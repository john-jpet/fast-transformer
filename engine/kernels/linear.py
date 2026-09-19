"""Warmup-selected BF16 skinny projections, with native cuBLAS as a candidate.

All products accumulate in FP32 and round once to BF16. Split-K uses a separate
FP32 reduction, never atomics. Selection is cached by tensor shape before the
decode graph is captured; prefill's large matrix products remain native.
"""

import ctypes
import statistics
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from kernels.pdl import wait as pdl_wait

from kernels.gemm import (
    _exact_gemm, _hoist_gemm, _hoist_trans_gemm, _persist_trans_gemm, _tma_gemm, _tmah_gemm, _tmap_gemm,
    _trans_gemm, exact_splits, persistent_programs,
)
from kernels.merged import Split
from kernels.tune import register


@triton.jit
def _gemv(
    x_ptr, weight_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr,
    CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pdl_wait()  # before any global memory access
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
    pdl_wait()  # before any global memory access
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
    pdl_wait()  # before any global memory access
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, BLOCK_S)
    values = tl.load(
        partial_ptr + splits[:, None] * COUNT + offsets[None, :],
        (splits[:, None] < SPLITS) & (offsets[None, :] < COUNT), other=0,
    )
    tl.store(out_ptr + offsets, tl.sum(values, axis=0), offsets < COUNT)


# --- Hopper TMA descriptor loads (Triton 3.1.0 experimental API) -------------
# ``tma`` is ``trans`` with the weight tile read through a tensor map. Every
# path fails closed: without a descriptor ``_project`` launches ``trans`` with
# the same constants (the same sums), and any exception retires the kind.
# ``tmah`` is ``tma`` with one x-tile load shared by four weight tiles per
# program; its fail-safe is ``_hoist_trans_gemm`` on the same grid, and it reads
# the same [64, 128] descriptor as ``tma`` (one descriptor per weight for both).
# ``tmap`` / ``tmap3`` are ``tma`` / ``tma3`` with one split on a persistent 1-D launch
# (kernels/gemm.py); fail-safe ``_persist_trans_gemm``, same grid, same BF16 output.
#: Kinds whose weights need a descriptor built in the eager pass.
TMA_KINDS = ("tma", "tma3", "tmah", "tmap", "tmap3")
#: Descriptor bytes; the blog's tuned value (the struct itself fits in 128).
TMA_SIZE = 512
_TMA_OFF = [False]
_TMA_DESCRIPTORS = {}
#: fill_2d_tma_descriptor returns Py_None without INCREF: never release a result.
_TMA_RESULTS = []

def _tma_retire(error):
    if not _TMA_OFF[0]:
        print(f"TMA descriptor loads retired: {error!r}", flush=True)
    _TMA_OFF[0] = True


def _tma_fill():
    if _TMA_OFF[0] or not hasattr(tl, "_experimental_descriptor_load"):
        return None
    return getattr(triton.runtime.driver.active.utils, "fill_2d_tma_descriptor", None)


def _tma_ready(weight, block_n, block_k):
    """Whether a descriptor of this weight's shape may be built.

    driver.c asserts on the encoder's result, which no ``except`` catches, so
    only whole, aligned BF16 tilings are ever submitted. (A child-process probe
    would cost an interpreter + torch import per workload under the sandboxed
    host: too much for the whole-run limit.)
    """
    try:
        n, k = weight.shape
        return (
            not _TMA_OFF[0] and weight.is_cuda and weight.dtype == torch.bfloat16 and weight.is_contiguous()
            and n % block_n == 0 and k % block_k == 0 and 2 * block_k >= 32
            and weight.data_ptr() % 128 == 0 and _tma_fill() is not None
        )
    except Exception as error:
        _tma_retire(error)
        return False


def _tma_descriptor(weight, block_n, block_k):
    """Device tensor map of ``weight`` tiled [block_n, block_k], or None. Built once, eagerly, kept alive."""
    try:
        n, k = weight.shape
        key = (weight.data_ptr(), n, k, block_n, block_k)
        desc = _TMA_DESCRIPTORS.get(key)
        if desc is not None or _TMA_OFF[0] or torch.cuda.is_current_stream_capturing():
            return desc
        if not _tma_ready(weight, block_n, block_k) or weight.data_ptr() % 128:
            return None
        fill = _tma_fill()
        host = torch.zeros(TMA_SIZE, dtype=torch.int8)
        buffer = (ctypes.c_char * TMA_SIZE).from_address(host.data_ptr())
        # (address, dim1, dim0, tile1, tile0, element bytes, host buffer): dim0 is the contiguous axis.
        _TMA_RESULTS.append(fill(weight.data_ptr(), n, k, block_n, block_k, weight.element_size(), buffer))
        desc = torch.empty(TMA_SIZE, dtype=torch.int8, device=weight.device)
        if desc.data_ptr() % 64:
            return None
        desc.copy_(host)
        torch.cuda.synchronize(weight.device)
        _TMA_DESCRIPTORS[key] = desc
        return desc
    except Exception as error:
        _tma_retire(error)
        return None


def _block_m(m):
    """Input-row lanes of a tile: tl.dot needs at least 16, and whole powers of two."""
    return 16 if m <= 16 else 32 if m <= 32 else 64


def _project(x, weight, config, split_ok=False, strict=False):
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
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=_block_m(m),
            num_warps=warps, num_stages=2,
        )
    elif kind == "hoist":
        # Four weight tiles per program share each x-tile load.
        block_m = _block_m(m)
        _hoist_gemm[(triton.cdiv(n, 4 * block_n), splits)](
            x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, TILES=4,
            EVEN_M=m == block_m, EVEN_N=n % (4 * block_n) == 0, EVEN_K=splits * chunk == k,
            WIDE=max(n * k, splits * m * n) + 8 * block_n * max(k, m) >= 2 ** 31,
            num_warps=warps, num_stages=2,
        )
    elif kind == "tmah":
        # "trans" orientation, four TMA weight tiles per program sharing each x-tile load.
        # The grid, the constants and ``partial`` are the same with or without a descriptor.
        block_m = _block_m(m)
        grid = (triton.cdiv(n, 4 * block_n), splits)
        even_n, even_k = n % (4 * block_n) == 0, splits * chunk == k
        wide = max(n * k, splits * m * n) + 8 * block_n * max(k, m) >= 2 ** 31
        launched = False
        desc = _tma_descriptor(weight, block_n, block_k) if even_n and even_k else None
        if desc is None and strict:
            raise RuntimeError("no TMA descriptor for this weight")
        if desc is not None:
            try:
                _tmah_gemm[grid](
                    x, desc, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
                    BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, TILES=4,
                    EVEN_M=m == block_m, WIDE=wide,
                    num_warps=warps, num_stages=2,
                )
                launched = True
            except Exception as error:
                _tma_retire(error)
                if strict:
                    raise
        if not launched:
            # No descriptor (capture before its eager pass, retired kind): the same sums, ordinary loads.
            _hoist_trans_gemm[grid](
                x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
                BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, TILES=4,
                EVEN_M=m == block_m, EVEN_N=even_n, EVEN_K=even_k, WIDE=wide,
                num_warps=warps, num_stages=2,
            )
    elif kind in ("tmap", "tmap3"):
        # Persistent launch: at most NUM_SMS programs, each walking its residue class of the
        # output tiles with the whole K loop inside. One split by construction: the FP32 sum is
        # rounded by the store into ``out``, so there is never a partial tensor (or a Split).
        if splits != 1:
            raise ValueError("persistent GEMM kinds take exactly one split")
        block_m = _block_m(m)
        tiles, steps = triton.cdiv(n, block_n), triton.cdiv(k, block_k)
        programs = persistent_programs(tiles)
        even_n, even_k = n % block_n == 0, k % block_k == 0
        wide = max(n * k, m * n) + 2 * block_n * max(k, m) >= 2 ** 31
        launched = False
        desc = _tma_descriptor(weight, block_n, block_k) if even_n and even_k else None
        if desc is None and strict:
            raise RuntimeError("no TMA descriptor for this weight")
        if desc is not None:
            try:
                _tmap_gemm[(programs,)](
                    x, desc, out, M=m, N=n, K=k, PROGRAMS=programs, N_TILES=tiles, STEPS=steps,
                    BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, EVEN_M=m == block_m, WIDE=wide,
                    # The stage count is the depth of the ring of prefetched weight tiles.
                    num_warps=warps, num_stages=3 if kind == "tmap3" else 2,
                )
                launched = True
            except Exception as error:
                _tma_retire(error)
                if strict:
                    raise
        if not launched:
            # No descriptor (capture before its eager pass, retired kind): the same sums, ordinary loads.
            _persist_trans_gemm[(programs,)](
                x, weight, out, M=m, N=n, K=k, PROGRAMS=programs, N_TILES=tiles, STEPS=steps,
                BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
                EVEN_M=m == block_m, EVEN_N=even_n, EVEN_K=even_k, WIDE=wide,
                num_warps=warps, num_stages=2,
            )
    else:
        # kernels/gemm.py: each mask exists only where that axis is ragged.
        block_m = _block_m(m)
        wide = max(n * k, splits * m * n) + 2 * block_n * max(k, m) >= 2 ** 31
        launched = False
        if kind in ("tma", "tma3"):
            desc = _tma_descriptor(weight, block_n, block_k) if n % block_n == 0 and splits * chunk == k else None
            if desc is None and strict:
                raise RuntimeError("no TMA descriptor for this weight")
            if desc is not None:
                try:
                    _tma_gemm[(n // block_n, splits)](
                        x, desc, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
                        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, EVEN_M=m == block_m, WIDE=wide,
                        # Descriptor loads are the only loads this compiler pipelines: the
                        # stage count is the depth of its ring of prefetched weight tiles.
                        num_warps=warps, num_stages=3 if kind == "tma3" else 2,
                    )
                    launched = True
                except Exception as error:
                    _tma_retire(error)
                    if strict:
                        raise
        # "tma" without a descriptor (capture before its eager pass, retired kind) is "trans": the same sums.
        kernel = None if launched else _exact_gemm if kind == "exact" else _trans_gemm
        if kernel is not None:
            kernel[(triton.cdiv(n, block_n), splits)](
                x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
                BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
                EVEN_M=m == block_m, EVEN_N=n % block_n == 0, EVEN_K=splits * chunk == k, WIDE=wide,
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


#: Verify blocks of up to 32 rows still read each weight once per step. (64-row
#: tiles were tried in candidate 68: their extra shapes and slow 64-lane
#: compiles pushed the whole run past the 900 s limit.)
MAX_ROWS = 32
_CHOICES = {}
_VALIDATED = {}
_TUNING_DEADLINE = None
_PROCESS_SECONDS = 28.0
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

    def hoisted(block_n, block_k, kind="hoist"):
        # Splits sized for programs of four tiles, then made exact.
        return (kind, block_n, block_k, exact_splits(k, block_k, gemm(4 * block_n, block_k)[3]), 4)

    configs = [gemm(64, 128)]
    if m == 1:
        configs += [("gemv", 8, 512, 1, 4), ("gemv", 16, 256, 1, 4)]
    elif m > 4:
        # Verify blocks fill most of the 16/32 input rows, so every tile reloads
        # a large x block: wider output tiles amortize it. Judged in the real
        # verify graph (DecodeState.refine), not only in isolation.
        tma = not _TMA_OFF[0] and n % 64 == 0 and k % 128 == 0
        if tma:
            # "trans" with TMA weight loads; offered only while every fail-safe
            # holds, and early: the per-shape budget drops the tail of this list.
            configs.append(tiled("tma", 64, 128))
            # "tma" with ONE split on a persistent launch of <= 132 programs, each
            # looping over its share of the tiles: no waves, no idle SMs, no FP32
            # partials; and the same three prefetched tiles deep (published
            # Hopper TMA configurations use 3-5).
            configs += [("tmap", 64, 128, 1, 4), ("tmap3", 64, 128, 1, 4)]
            if n % (4 * 64) == 0:
                # "tma" with the x-tile load hoisted over four weight tiles per program.
                configs.append(hoisted(64, 128, "tmah"))
            # (tma3 and the ordinary hoist are left out while TMA is on: the
            # ~6 s per shape cover about five layouts under the sandboxed host.)
        else:
            configs.append(hoisted(64, 128))
        if not tma:
            # "tma" is "trans" plus TMA loads: only one of the two is ever listed. "exact" (the
            # narrower matrix-multiply fragments, and "hoist" without its shared x load) makes
            # room for the persistent kinds whenever TMA is on: ~6 s per shape is 6-7 layouts.
            configs += [tiled("exact", 64, 128), tiled("trans", 64, 128)]
    return configs


def _agrees(probe, weight, config, reference):
    """Operator sanity check; a layout that fails to compile or launch is simply not offered."""
    try:
        # Split layouts are checked through their FP32 partials: the verify
        # path never launches the merge kernel, so tuning need not compile it.
        actual = _project(probe, weight, config, split_ok=True, strict=True)
        actual = actual.partial.sum(0) if isinstance(actual, Split) else actual.float()
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
    if m <= 4 or m > 32:
        # Above 32 rows cuBLAS is a strong default: only a timed layout replaces it.
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


def _choose(x, weight, split_ok=False):
    global _TUNING_DEADLINE
    if x.shape[0] > 4 and not _TMA_OFF[0]:
        # The child-process probe (seconds) stays outside the tuning clock.
        _tma_ready(weight, 64, 128)
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
        elapsed = _cold_graph_time(lambda: _project(x, weight, config, split_ok), flush)
        validated.append((elapsed, config))
        if elapsed < best_ms * 0.985:
            best_ms, best = elapsed, config
    if best is not None:
        # Recheck after compilation/tuning so GPU clock ramp-up cannot make a
        # later candidate look faster than an initially cold native baseline.
        native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
        best_ms = _cold_graph_time(lambda: _project(x, weight, best, split_ok), flush)
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
        # Timed the way this caller will use it: with or without the merge launch.
        _CHOICES[key] = _choose(flat, weight, split_ok)
        # None is cuBLAS. The captured decode step re-judges these layouts.
        # Refinement order = weight traffic per step: the vocabulary projection
        # runs once, every other projection once per layer (36 here).
        traffic = weight.shape[0] * weight.shape[1] // (36 if weight.shape[0] > 65536 else 1)
        register(
            ("projection",) + key[1:], rows, traffic,
            # The three fastest in isolation: refinement time is scarce, and a
            # layout that was far behind alone has not won inside the graph.
            [config for _, config in sorted(_VALIDATED.get(key, ()), key=lambda item: item[0])][:3],
            lambda: _CHOICES[key], lambda config: _CHOICES.__setitem__(key, config),
        )
    choice = _CHOICES[key]
    if rows > 4 and not _TMA_OFF[0] and any(config and config[0] in TMA_KINDS for _, config in _VALIDATED.get(key, ())):
        # Every weight of a shape that validated a TMA kind gets its descriptor in the
        # eager pass, so the captured step can re-judge the kind for all layers.
        _tma_descriptor(weight, 64, 128)
    if choice is None:
        return F.linear(x, weight)
    result = _project(flat, weight, choice, split_ok)
    if isinstance(result, Split):
        result.shape = (*x.shape[:-1], weight.shape[0])
        return result
    return result.reshape(*x.shape[:-1], weight.shape[0])
