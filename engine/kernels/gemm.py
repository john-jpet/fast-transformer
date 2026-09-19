"""Mask-free skinny GEMM tiles for kernels.linear: the same FP32-accumulated
BF16 product as ``_skinny_gemm``, partitioned so no program or lane is idle.

``exact`` compiles each mask in only when that axis is ragged and expects a
split count that tiles K with whole BLOCK_K blocks (no dead split programs).
``trans`` computes the transposed tile product w[N', K'] @ x^T[K', M'] so the
64-row first operand selects the wider matrix-multiply fragments. Both only
reorder the FP32 sum over K; the single rounding to BF16 is unchanged.
``hoist`` is ``exact`` with one x-tile load shared by TILES weight tiles: with
16-32 input rows a program's x chunk is a quarter to a half of the bytes of a
64-row weight tile, and every tile used to reload it. Same sums as ``exact``.
``tmah`` is that hoist in ``trans`` orientation with TMA weight loads (``tma``
x ``hoist``); ``_hoist_trans_gemm`` is its ordinary-load twin on the same grid.
``tmap`` is ``tma`` with SPLITS == 1 on a PERSISTENT 1-D launch of at most NUM_SMS
programs, each looping over its share of the output tiles (no scheduling waves, no
idle SMs, no FP32 partials); ``_persist_trans_gemm`` is its ordinary-load twin.
"""

import triton
import triton.language as tl

from kernels.pdl import wait as pdl_wait


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
    pdl_wait()  # before any global memory access
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
    pdl_wait()  # before any global memory access
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



@triton.jit
def _hoist_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    pdl_wait()  # before any global memory access
    rows = tl.arange(0, BLOCK_M)
    columns = tl.program_id(0) * (TILES * BLOCK_N) + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    k = split * CHUNK + tl.arange(0, BLOCK_K)
    row_ok = rows[:, None] < M
    x_ptrs = x_ptr + rows[:, None] * K + k[None, :]
    # Tile t covers columns + t * BLOCK_N: its weights start t * BLOCK_N rows further.
    w_ptrs = weight_ptr + columns[None, :] * K + k[:, None]
    w_step = BLOCK_N * K
    ok0 = columns[None, :] < N
    ok1 = (columns[None, :] + BLOCK_N) < N
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if TILES == 4:
        ok2 = (columns[None, :] + 2 * BLOCK_N) < N
        ok3 = (columns[None, :] + 3 * BLOCK_N) < N
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        k_ok = (k + step * BLOCK_K) < K
        x = _load_tile(x_ptrs, row_ok, k_ok[None, :], EVEN_M, EVEN_K)
        acc0 = tl.dot(x, _load_tile(w_ptrs, k_ok[:, None], ok0, EVEN_K, EVEN_N), acc0)
        acc1 = tl.dot(x, _load_tile(w_ptrs + w_step, k_ok[:, None], ok1, EVEN_K, EVEN_N), acc1)
        if TILES == 4:
            acc2 = tl.dot(x, _load_tile(w_ptrs + 2 * w_step, k_ok[:, None], ok2, EVEN_K, EVEN_N), acc2)
            acc3 = tl.dot(x, _load_tile(w_ptrs + 3 * w_step, k_ok[:, None], ok3, EVEN_K, EVEN_N), acc3)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    o_ptrs = out_ptr + split * (M * N) + rows[:, None] * N + columns[None, :]
    _store_tile(o_ptrs, acc0, row_ok, ok0, EVEN_M, EVEN_N)
    _store_tile(o_ptrs + BLOCK_N, acc1, row_ok, ok1, EVEN_M, EVEN_N)
    if TILES == 4:
        _store_tile(o_ptrs + 2 * BLOCK_N, acc2, row_ok, ok2, EVEN_M, EVEN_N)
        _store_tile(o_ptrs + 3 * BLOCK_N, acc3, row_ok, ok3, EVEN_M, EVEN_N)


@triton.jit
def _tma_gemm(
    x_ptr, desc_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, WIDE: tl.constexpr,
):
    """``_trans_gemm`` with the weight tile read through a Hopper TMA tensor map.

    ``desc_ptr`` is the device copy of a 2-D descriptor of weight[N, K] tiled
    [BLOCK_N, BLOCK_K]; the load takes the ELEMENT offsets (row, k) of a whole
    tile and returns it in storage orientation, which is ``trans``'s first
    operand: same ``tl.dot(weight, x^T, acc)``, same K order, so the same bits
    as ``trans`` when the load returns the same bytes. (Transposing the loaded
    tile into ``exact``'s ``tl.dot(x, weight)`` segfaults the Triton 3.1.0
    compiler for cuda:90 - never do that.) Loads only: descriptor stores are
    nondeterministic (Triton #6638). The launcher guarantees N % BLOCK_N == 0
    and SPLITS * CHUNK == K, so no tile reaches out of bounds.
    """
    pdl_wait()  # before any global memory access
    rows = tl.arange(0, BLOCK_M)
    first = tl.program_id(0) * BLOCK_N
    columns = first + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    start = split * CHUNK
    k = start + tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
    acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        weight = tl._experimental_descriptor_load(
            desc_ptr, [first, start + step * BLOCK_K], [BLOCK_N, BLOCK_K], tl.bfloat16,
        )
        x = _load_tile(x_ptrs, row_ok, row_ok, True, EVEN_M)
        acc = tl.dot(weight, x, acc)
        x_ptrs += BLOCK_K
    # acc[j, i] is output row i, column columns[j].
    _store_tile(
        out_ptr + split * (M * N) + columns[:, None] + rows[None, :] * N, acc,
        row_ok, row_ok, True, EVEN_M,
    )


@triton.jit
def _hoist_trans_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    """``_trans_gemm`` with one x-tile load shared by TILES weight tiles per program.

    The ordinary-load twin of ``_tmah_gemm``: same grid, same partial layout,
    same ``tl.dot(weight_t, x^T, acc_t)`` per K chunk, so the launcher may fall
    back to it without changing anything downstream.
    """
    pdl_wait()  # before any global memory access
    rows = tl.arange(0, BLOCK_M)
    columns = tl.program_id(0) * (TILES * BLOCK_N) + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    k = split * CHUNK + tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    # weight tiles [BLOCK_N, BLOCK_K] in storage orientation; the shared x tile transposed [BLOCK_K, BLOCK_M].
    w_ptrs = weight_ptr + columns[:, None] * K + k[None, :]
    x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
    # Tile t covers columns + t * BLOCK_N: its weights start t * BLOCK_N rows further.
    w_step = BLOCK_N * K
    ok0 = columns[:, None] < N
    ok1 = (columns[:, None] + BLOCK_N) < N
    acc0 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    if TILES == 4:
        ok2 = (columns[:, None] + 2 * BLOCK_N) < N
        ok3 = (columns[:, None] + 3 * BLOCK_N) < N
        acc2 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        acc3 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        k_ok = (k + step * BLOCK_K) < K
        x = _load_tile(x_ptrs, k_ok[:, None], row_ok, EVEN_K, EVEN_M)
        acc0 = tl.dot(_load_tile(w_ptrs, ok0, k_ok[None, :], EVEN_N, EVEN_K), x, acc0)
        acc1 = tl.dot(_load_tile(w_ptrs + w_step, ok1, k_ok[None, :], EVEN_N, EVEN_K), x, acc1)
        if TILES == 4:
            acc2 = tl.dot(_load_tile(w_ptrs + 2 * w_step, ok2, k_ok[None, :], EVEN_N, EVEN_K), x, acc2)
            acc3 = tl.dot(_load_tile(w_ptrs + 3 * w_step, ok3, k_ok[None, :], EVEN_N, EVEN_K), x, acc3)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    # acc_t[j, i] is output row i, column columns[j] + t * BLOCK_N.
    o_ptrs = out_ptr + split * (M * N) + columns[:, None] + rows[None, :] * N
    _store_tile(o_ptrs, acc0, ok0, row_ok, EVEN_N, EVEN_M)
    _store_tile(o_ptrs + BLOCK_N, acc1, ok1, row_ok, EVEN_N, EVEN_M)
    if TILES == 4:
        _store_tile(o_ptrs + 2 * BLOCK_N, acc2, ok2, row_ok, EVEN_N, EVEN_M)
        _store_tile(o_ptrs + 3 * BLOCK_N, acc3, ok3, row_ok, EVEN_N, EVEN_M)


@triton.jit
def _tmah_gemm(
    x_ptr, desc_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    TILES: tl.constexpr,
    EVEN_M: tl.constexpr, WIDE: tl.constexpr,
):
    """``_hoist_trans_gemm`` with every weight tile read through the TMA tensor map.

    One ordinary x-tile load per K chunk feeds TILES descriptor loads, each at
    its own element offset (first + t * BLOCK_N, k): ``_tma_gemm``'s descriptor
    (the same [BLOCK_N, BLOCK_K] tiling), ``_tma_gemm``'s operand orientation
    (never transpose a loaded tile: see there), ``_trans_gemm``'s sums. The
    launcher guarantees N % (TILES * BLOCK_N) == 0 and SPLITS * CHUNK == K.
    """
    pdl_wait()  # before any global memory access
    rows = tl.arange(0, BLOCK_M)
    first = tl.program_id(0) * (TILES * BLOCK_N)
    columns = first + tl.arange(0, BLOCK_N)
    if WIDE:
        columns = columns.to(tl.int64)
    split = tl.program_id(1)
    start = split * CHUNK
    k = start + tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
    acc0 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    if TILES == 4:
        acc2 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        acc3 = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for step in range(0, CHUNK // BLOCK_K):
        at = start + step * BLOCK_K
        x = _load_tile(x_ptrs, row_ok, row_ok, True, EVEN_M)
        acc0 = tl.dot(tl._experimental_descriptor_load(
            desc_ptr, [first, at], [BLOCK_N, BLOCK_K], tl.bfloat16), x, acc0)
        acc1 = tl.dot(tl._experimental_descriptor_load(
            desc_ptr, [first + BLOCK_N, at], [BLOCK_N, BLOCK_K], tl.bfloat16), x, acc1)
        if TILES == 4:
            acc2 = tl.dot(tl._experimental_descriptor_load(
                desc_ptr, [first + 2 * BLOCK_N, at], [BLOCK_N, BLOCK_K], tl.bfloat16), x, acc2)
            acc3 = tl.dot(tl._experimental_descriptor_load(
                desc_ptr, [first + 3 * BLOCK_N, at], [BLOCK_N, BLOCK_K], tl.bfloat16), x, acc3)
        x_ptrs += BLOCK_K
    # acc_t[j, i] is output row i, column columns[j] + t * BLOCK_N.
    o_ptrs = out_ptr + split * (M * N) + columns[:, None] + rows[None, :] * N
    _store_tile(o_ptrs, acc0, row_ok, row_ok, True, EVEN_M)
    _store_tile(o_ptrs + BLOCK_N, acc1, row_ok, row_ok, True, EVEN_M)
    if TILES == 4:
        _store_tile(o_ptrs + 2 * BLOCK_N, acc2, row_ok, row_ok, True, EVEN_M)
        _store_tile(o_ptrs + 3 * BLOCK_N, acc3, row_ok, row_ok, True, EVEN_M)


#: Streaming multiprocessors of the H100: a launch of at most this many programs is one
#: scheduling wave with no idle SM; more is cut into waves, fewer leaves SMs idle.
NUM_SMS = 132


def persistent_programs(tiles, sms=NUM_SMS):
    """Programs of a persistent launch over ``tiles`` output tiles: at most ``sms``, evenly loaded.

    Every program walks ceil(tiles / sms) tiles at most, so that depth is fixed; the
    fewest programs that still reach it keep every program busy to the end (304 tiles:
    102 programs x 3 rather than 40 x 3 + 92 x 2; 2374 tiles: 132 x 18; <= 132: one each).
    """
    depth = -(-tiles // sms)
    return -(-tiles // depth)


@triton.jit
def _persist_trans_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    PROGRAMS: tl.constexpr, N_TILES: tl.constexpr, STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr, WIDE: tl.constexpr,
):
    """``_trans_gemm`` with SPLITS == 1 on a persistent 1-D grid: the ordinary-load twin of ``_tmap_gemm``.

    Program p of PROGRAMS owns output tiles p, p + PROGRAMS, p + 2 PROGRAMS, ... (disjoint
    and covering: each tile index has one residue). For each tile the INNER loop runs the
    whole K axis, STEPS = ceil(K / BLOCK_K) chunks in ascending order, as
    ``acc = tl.dot(weight[BLOCK_N, BLOCK_K], x^T[BLOCK_K, BLOCK_M], acc)`` with an FP32
    accumulator that starts at zero: operand shapes, orientation and chunk order are those
    of ``_trans_gemm`` with SPLITS == 1 (CHUNK == STEPS * BLOCK_K), so each output element
    is the same FP32 sum, and it is rounded once, by the store into the BF16 output [M, N].
    There is no FP32 partial tensor. Nothing is carried from tile to tile: every pointer is
    rebuilt from the tile index (a pass-through loop-carried value disables pipelining).
    """
    pdl_wait()  # before any global memory access
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    for tile in range(pid, N_TILES, PROGRAMS):
        columns = tile * BLOCK_N + lanes
        if WIDE:
            columns = columns.to(tl.int64)
        col_ok = columns[:, None] < N
        # weight tile [BLOCK_N, BLOCK_K] in storage orientation; x tile transposed [BLOCK_K, BLOCK_M].
        w_ptrs = weight_ptr + columns[:, None] * K + k[None, :]
        x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for step in range(0, STEPS):
            k_ok = (k + step * BLOCK_K) < K
            weight = _load_tile(w_ptrs, col_ok, k_ok[None, :], EVEN_N, EVEN_K)
            x = _load_tile(x_ptrs, k_ok[:, None], row_ok, EVEN_K, EVEN_M)
            acc = tl.dot(weight, x, acc)
            w_ptrs += BLOCK_K
            x_ptrs += BLOCK_K
        # acc[j, i] is output row i, column columns[j].
        _store_tile(out_ptr + columns[:, None] + rows[None, :] * N, acc, col_ok, row_ok, EVEN_N, EVEN_M)


@triton.jit
def _tmap_gemm(
    x_ptr, desc_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    PROGRAMS: tl.constexpr, N_TILES: tl.constexpr, STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr, WIDE: tl.constexpr,
):
    """``_persist_trans_gemm`` with the weight tile read through the TMA tensor map.

    The K-chunk loop is the INNERMOST ``for`` (only that loop is pipelined: its
    descriptor loads become a ring of ``num_stages`` prefetched tiles); the tile loop is
    outside it and carries nothing. ``_tma_gemm``'s descriptor ([BLOCK_N, BLOCK_K] tiling
    of weight[N, K]), its operand orientation (the loaded tile is the FIRST operand of
    ``tl.dot``; never transpose it: see ``_tma_gemm``) and ``_trans_gemm``'s sums with
    SPLITS == 1. The launcher guarantees N == N_TILES * BLOCK_N and K == STEPS * BLOCK_K.
    """
    pdl_wait()  # before any global memory access
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    row_ok = rows[None, :] < M
    for tile in range(pid, N_TILES, PROGRAMS):
        first = tile * BLOCK_N
        columns = first + lanes
        if WIDE:
            columns = columns.to(tl.int64)
        x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
        for step in range(0, STEPS):
            weight = tl._experimental_descriptor_load(
                desc_ptr, [first, step * BLOCK_K], [BLOCK_N, BLOCK_K], tl.bfloat16,
            )
            x = _load_tile(x_ptrs, row_ok, row_ok, True, EVEN_M)
            acc = tl.dot(weight, x, acc)
            x_ptrs += BLOCK_K
        # acc[j, i] is output row i, column columns[j].
        _store_tile(out_ptr + columns[:, None] + rows[None, :] * N, acc, row_ok, row_ok, True, EVEN_M)


def exact_splits(k, block_k, wanted):
    """Largest split count <= wanted whose chunks tile K with whole BLOCK_K blocks."""
    if k % block_k:
        return wanted
    blocks = k // block_k
    return max(splits for splits in range(1, wanted + 1) if blocks % splits == 0)
