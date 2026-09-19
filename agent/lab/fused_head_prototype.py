import triton
import triton.language as tl
from kernels.gemm import _load_tile

@triton.jit
def _head_tiles(
    x_ptr, weight_ptr, best_value, best_index,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    tl.static_assert(SPLITS == 1)
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
    # Match materialized BF16 logits BEFORE selecting a winner.
    logits = tl.where(col_ok, acc.to(tl.bfloat16).to(tl.float32), -float("inf"))
    maximum, local = tl.max(logits, axis=1, return_indices=True, return_indices_tie_break_left=True)
    tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    tl.store(best_value + rows * tiles + tile, maximum, rows < M)
    tl.store(best_index + rows * tiles + tile, tile * BLOCK_N + local, rows < M)
