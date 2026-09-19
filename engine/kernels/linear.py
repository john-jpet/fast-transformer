"""Warmup-selected BF16 skinny projections, with native cuBLAS as a candidate.

All products accumulate in FP32 and round once to BF16. Split-K uses a separate
FP32 reduction, never atomics. Selection is cached by tensor shape before the
decode graph is captured; prefill's large matrix products remain native.
"""

import statistics
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from kernels.gemm import _exact_gemm, _trans_gemm, exact_splits
from kernels.merged import Split
from kernels.tune import register


@triton.jit
def _gemv(
    x_ptr, weight_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr,
    CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    columns = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(split * CHUNK, (split + 1) * CHUNK, BLOCK_K):
        k = start + columns
        x = tl.load(x_ptr + k, k < K, other=0).to(tl.float32)
        weight = tl.load(
            weight_ptr + rows[:, None] * K + k[None, :],
            (rows[:, None] < N) & (k[None, :] < K), other=0,
        ).to(tl.float32)
        acc += weight * x[None, :]
    result = tl.sum(acc, axis=1)
    tl.store(out_ptr + split * N + rows, result, rows < N)


@triton.jit
def _skinny_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr = 16,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    reduction = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(split * CHUNK, (split + 1) * CHUNK, BLOCK_K):
        k = start + reduction
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K), other=0,
        )
        weight = tl.load(
            weight_ptr + columns[None, :] * K + k[:, None],
            (columns[None, :] < N) & (k[:, None] < K), other=0,
        )
        acc = tl.dot(x, weight, acc)
    tl.store(
        out_ptr + split * M * N + rows[:, None] * N + columns[None, :], acc,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def _merge_projection(
    partial_ptr, out_ptr, COUNT: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, BLOCK_S)
    values = tl.load(
        partial_ptr + splits[:, None] * COUNT + offsets[None, :],
        (splits[:, None] < SPLITS) & (offsets[None, :] < COUNT), other=0,
    )
    tl.store(out_ptr + offsets, tl.sum(values, axis=0), offsets < COUNT)


def _project(x, weight, config, split_ok=False):
    kind, block_n, block_k, splits, warps = config
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = out if splits == 1 else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    chunk = triton.cdiv(k, splits * block_k) * block_k
    if kind == "gemv":
        _gemv[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps,
        )
    elif kind == "gemm":
        _skinny_gemm[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=16 if m <= 16 else 32,
            num_warps=warps, num_stages=2,
        )
    else:
        # kernels/gemm.py: each mask exists only where that axis is ragged.
        block_m = 16 if m <= 16 else 32
        (_exact_gemm if kind == "exact" else _trans_gemm)[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
            EVEN_M=m == block_m, EVEN_N=n % block_n == 0, EVEN_K=splits * chunk == k,
            WIDE=max(n * k, splits * m * n) + 2 * block_n * max(k, m) >= 2 ** 31,
            num_warps=warps, num_stages=2,
        )
    if splits > 1 and split_ok:
        # The consumer kernel sums and rounds the partials itself.
        return Split(partial, (m, n))
    if splits > 1:
        _merge_projection[(triton.cdiv(m * n, 512),)](
            partial, out, COUNT=m * n, SPLITS=splits,
            BLOCK_S=triton.next_power_of_2(splits), BLOCK=512, num_warps=4,
        )
    return out


def _cold_graph_time(fn, flush):
    # CPU dispatch is absent from decode. Benchmark a graph, with >H100 L2
    # bytes touched before each projection, to approximate its cold weights.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            flush.zero_()
            fn()
    graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 8)
    return statistics.median(times)


#: Verify blocks of up to 32 rows still read each weight once per step.
MAX_ROWS = 32
_CHOICES = {}
_VALIDATED = {}
_TUNING_DEADLINE = None
_PROCESS_SECONDS = 18.0
_SHAPE_SECONDS = 6.0


def _candidates(m, n, k):
    """Official runs: wider tile searches, lossless 12-bit planes and word-sized
    loads never beat these in the captured step, so keep the list short."""
    def gemm(block_n, block_k):
        splits = min(8, triton.next_power_of_2(triton.cdiv(512, triton.cdiv(n, block_n))))
        return ("gemm", block_n, block_k, splits, 4)

    def tiled(kind, block_n, block_k):
        # Whole-block splits: K=2560 takes 5 where the power of two left one
        # of 8 split programs entirely masked; K=9728 takes 4.
        return (kind, block_n, block_k, exact_splits(k, block_k, gemm(block_n, block_k)[3]), 4)

    configs = [gemm(64, 128)]
    if m == 1:
        configs += [("gemv", 8, 512, 1, 4), ("gemv", 16, 256, 1, 4)]
    elif m > 4:
        # Verify blocks fill most of the 16/32 input rows, so every tile reloads
        # a large x block: wider output tiles amortize it. Judged in the real
        # verify graph (DecodeState.refine), not only in isolation.
        configs += [tiled("exact", 64, 128), tiled("trans", 64, 128), gemm(256, 128)]
    return configs


def _agrees(probe, weight, config, reference):
    """Operator sanity check; a layout that fails to compile or launch is simply not offered."""
    try:
        actual = _project(probe, weight, config).float()
        return bool(((actual - reference.float()).abs() <= reference.float().abs() * 0.016 + 0.001).all())
    except Exception as error:
        print(f"BF16 projection layout {config} skipped: {error!r}", flush=True)
        return False


def _inherit(x, weight):
    """Past the budget, a verify block takes the layout measured for this weight at another row count.

    The second block size tried at warmup would otherwise run cuBLAS-only and
    lose the comparison for that reason alone. The verify-graph refinement
    still re-judges the layout against cuBLAS if this block size wins.
    """
    m, k = x.shape
    n = weight.shape[0]
    if m <= 4:
        return None
    for (device, rows, width, inner), config in list(_CHOICES.items()):
        if config is None or device != x.device or rows <= 4 or (width, inner) != (n, k):
            continue
        generator = torch.Generator(device=x.device).manual_seed(1729)
        probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        if not _agrees(probe, weight, config, F.linear(probe, weight)):
            continue
        _VALIDATED[(x.device, m, n, k)] = [(0.0, config), (1.0, None)]
        return config
    return None


def _choose(x, weight):
    global _TUNING_DEADLINE
    now = time.monotonic()
    if _TUNING_DEADLINE is None:
        _TUNING_DEADLINE = now + _PROCESS_SECONDS
    if now >= _TUNING_DEADLINE:
        return _inherit(x, weight)
    # Every workload is a fresh process: bound each shape and the process so
    # compilation fits the load/warmup and whole-run budgets, and so one slow
    # shape cannot leave the later projections unmeasured.
    shape_deadline = min(_TUNING_DEADLINE, now + _SHAPE_SECONDS)
    m, k = x.shape
    n = weight.shape[0]
    configs = _candidates(m, n, k)
    # Private generator: tuning must not change any caller's RNG state.
    generator = torch.Generator(device=x.device).manual_seed(1729)
    probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    reference = F.linear(probe, weight)
    flush = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.int32)
    native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
    best_ms, best = native_ms, None
    validated = [(native_ms, None)]
    _VALIDATED[(x.device, m, n, k)] = validated
    for config in configs:
        if time.monotonic() >= shape_deadline:
            break
        # Reject a kernel that fails an operator sanity check. Full-model
        # correctness still comes from the platform's own-prefix replay.
        if not _agrees(probe, weight, config, reference):
            continue
        elapsed = _cold_graph_time(lambda: _project(x, weight, config), flush)
        validated.append((elapsed, config))
        if elapsed < best_ms * 0.985:
            best_ms, best = elapsed, config
    if best is not None:
        # Recheck after compilation/tuning so GPU clock ramp-up cannot make a
        # later candidate look faster than an initially cold native baseline.
        native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
        best_ms = _cold_graph_time(lambda: _project(x, weight, best), flush)
        if best_ms >= native_ms * 0.985:
            best_ms, best = native_ms, None
    print(f"BF16 projection warmup: backend={best or 'cublas'} cold_graph_ratio={best_ms / native_ms:.3f}", flush=True)
    return best


def keep_native(rows, weight):
    """Leave a shape on cuBLAS without spending tuning budget on it."""
    _CHOICES.setdefault((weight.device, rows, weight.shape[0], weight.shape[1]), None)


def linear(x, weight, split_ok=False):
    """x @ weight.T in BF16. With ``split_ok`` the result may be a ``Split`` for a consumer kernel."""
    rows = x.numel() // x.shape[-1]
    if rows > MAX_ROWS or x.dtype != torch.bfloat16 or not weight.is_contiguous():
        return F.linear(x, weight)
    flat = x.reshape(rows, x.shape[-1]).contiguous()
    key = (x.device, rows, weight.shape[0], weight.shape[1])
    if key not in _CHOICES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("projection selection must finish during eager warmup")
        _CHOICES[key] = _choose(flat, weight)
        # None is cuBLAS. The captured decode step re-judges these layouts.
        # Refinement order = weight traffic per step: the vocabulary projection
        # runs once, every other projection once per layer (36 here).
        traffic = weight.shape[0] * weight.shape[1] // (36 if weight.shape[0] > 65536 else 1)
        register(
            ("projection",) + key[1:], rows, traffic,
            [config for _, config in sorted(_VALIDATED.get(key, ()), key=lambda item: item[0])],
            lambda: _CHOICES[key], lambda config: _CHOICES.__setitem__(key, config),
        )
    choice = _CHOICES[key]
    if choice is None:
        return F.linear(x, weight)
    result = _project(flat, weight, choice, split_ok)
    if isinstance(result, Split):
        result.shape = (*x.shape[:-1], weight.shape[0])
        return result
    return result.reshape(*x.shape[:-1], weight.shape[0])
