"""Mask-free skinny GEMM tiles for kernels.linear: the same FP32-accumulated
BF16 product as ``_skinny_gemm``, partitioned so no program or lane is idle.

``exact`` compiles each mask in only when that axis is ragged and expects a
split count that tiles K with whole BLOCK_K blocks (no dead split programs).
``trans`` computes the transposed tile product w[N', K'] @ x^T[K', M'] so the
64-row first operand selects the wider matrix-multiply fragments. Both only
reorder the FP32 sum over K; the single rounding to BF16 is unchanged.
"""

import triton
import triton.language as tl


@triton.jit
def _load_tile(ptrs, first_ok, second_ok, EVEN_FIRST: tl.constexpr, EVEN_SECOND: tl.constexpr):
    """Load a 2-D tile; each mask is compiled in only when that axis is ragged."""
    if EVEN_FIRST and EVEN_SECOND:
        tile = tl.load(ptrs)
    elif EVEN_SECOND:
        tile = tl.load(ptrs, first_ok, other=0)
    elif EVEN_FIRST:
        tile = tl.load(ptrs, second_ok, other=0)
    else:
        tile = tl.load(ptrs, first_ok & second_ok, other=0)
    return tile


@triton.jit
def _store_tile(ptrs, value, first_ok, second_ok, EVEN_FIRST: tl.constexpr, EVEN_SECOND: tl.constexpr):
    if EVEN_FIRST and EVEN_SECOND:
        tl.store(ptrs, value)
    elif EVEN_SECOND:
        tl.store(ptrs, value, first_ok)
    elif EVEN_FIRST:
        tl.store(ptrs, value, second_ok)
    else:
        tl.store(ptrs, value, first_ok & second_ok)


@triton.jit
def _exact_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    k = split * CHUNK + tl.arange(0, BLOCK_K)
    row_ok = rows[:, None] < M
    col_ok = columns[None, :] < N
    x_ptrs = x_ptr + rows[:, None] * K + k[None, :]
    w_ptrs = weight_ptr + columns[None, :] * K + k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        k_ok = (k + step * BLOCK_K) < K
        x = _load_tile(x_ptrs, row_ok, k_ok[None, :], EVEN_M, EVEN_K)
        weight = _load_tile(w_ptrs, k_ok[:, None], col_ok, EVEN_K, EVEN_N)
        acc = tl.dot(x, weight, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    _store_tile(
        out_ptr + split * (M * N) + rows[:, None] * N + columns[None, :], acc,
        row_ok, col_ok, EVEN_M, EVEN_N,
    )


@triton.jit
def _trans_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    k = split * CHUNK + tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    col_ok = columns[:, None] < N
    # weight tile [BLOCK_N, BLOCK_K] in storage orientation; x tile transposed [BLOCK_K, BLOCK_M].
    w_ptrs = weight_ptr + columns[:, None] * K + k[None, :]
    x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
    acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        k_ok = (k + step * BLOCK_K) < K
        weight = _load_tile(w_ptrs, col_ok, k_ok[None, :], EVEN_N, EVEN_K)
        x = _load_tile(x_ptrs, k_ok[:, None], row_ok, EVEN_K, EVEN_M)
        acc = tl.dot(weight, x, acc)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    # acc[j, i] is output row i, column columns[j].
    _store_tile(
        out_ptr + split * (M * N) + columns[:, None] + rows[None, :] * N, acc,
        col_ok, row_ok, EVEN_N, EVEN_M,
    )



def exact_splits(k, block_k, wanted):
    """Largest split count <= wanted whose chunks tile K with whole BLOCK_K blocks."""
    if k % block_k:
        return wanted
    blocks = k // block_k
    return max(splits for splits in range(1, wanted + 1) if blocks % splits == 0)
