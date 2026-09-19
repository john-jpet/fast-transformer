"""Prefill gate/up projection with the SwiGLU epilogue in the same kernel.

Ported from john-jpet's fork of this engine (team dryfter). A paired Triton
GEMM computes the gate and up tiles of one output block and writes only the
activated product, keeping the native rounding boundaries (both projections
round to BF16, SiLU rounds to BF16, then the product rounds to BF16). That
saves the separate SwiGLU pass over [rows, 2*I]. Warmup times it against
cuBLAS + SwiGLU and keeps it only if it is faster; capture never compiles,
times or changes the choice. Decode blocks (<= 32 rows) do not come here: their
split GEMM already hands FP32 partials to SwiGLU (kernels/merged.py).
"""

import time

import torch
import triton
import triton.language as tl

from kernels.linear import _cold_graph_time, linear
from kernels.swiglu import swiglu


@triton.jit
def _activate(gate, up):
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    return (activated * up).to(tl.bfloat16)


@triton.jit
def _paired_projection(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Group adjacent row tiles so neighbouring programs reuse weight tiles.
    pid = tl.program_id(0).to(tl.int64)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(I, BN)
    group = pid // (8 * tiles_n)
    first_m = group * 8
    group_m = tl.minimum(tiles_m - first_m, 8)
    within = pid % (8 * tiles_n)
    tile_m = first_m + within % group_m
    tile_n = within // group_m
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(x_ptr + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        wg = tl.load(weight_ptr + cols[None, :] * K + k[:, None],
                     (cols[None, :] < I) & (k[:, None] < K), other=0)
        wu = tl.load(weight_ptr + (I + cols[None, :]) * K + k[:, None],
                     (cols[None, :] < I) & (k[:, None] < K), other=0)
        gate = tl.dot(x, wg, gate)
        up = tl.dot(x, wu, up)
    tl.store(out_ptr + rows[:, None] * I + cols[None, :], _activate(gate, up),
             (rows[:, None] < M) & (cols[None, :] < I))


def _run(x, weight, option):
    """Contiguous BF16 x[M,K], packed weights[2*I,K] -> new BF16 [M,I]."""
    if option is None:
        return swiglu(linear(x, weight))
    m, k = x.shape
    width = weight.shape[0] // 2
    out = torch.empty((m, width), device=x.device, dtype=x.dtype)
    bm, bn, bk, warps = option
    _paired_projection[(triton.cdiv(m, bm) * triton.cdiv(width, bn),)](
        x, weight, out, M=m, I=width, K=k, BM=bm, BN=bn, BK=bk,
        num_warps=warps, num_stages=3,
    )
    return out


_CHOICES = {}
_DEADLINE = None


def _choose(x, weight):
    global _DEADLINE
    now = time.monotonic()
    if _DEADLINE is None:
        _DEADLINE = now + 6.0
    if now >= _DEADLINE:
        return None
    # Operator checks use nonzero values, never the zero capture-warmup input.
    generator = torch.Generator(device=x.device).manual_seed(31415)
    probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    reference = _run(probe, weight, None)
    flush = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.int32)
    native_ms = _cold_graph_time(lambda: _run(x, weight, None), flush)
    best, best_ms = None, native_ms
    for option in ((32, 64, 32, 4), (64, 64, 32, 8)):
        if time.monotonic() >= _DEADLINE:
            break
        try:
            actual = _run(probe, weight, option)
        except Exception as error:  # a layout that cannot compile is not a candidate
            print(f"gated projection layout {option} skipped: {error!r}", flush=True)
            continue
        # A gross-error screen; the teacher-forced judge decides correctness.
        if not torch.allclose(actual.float(), reference.float(), rtol=0.035, atol=0.02):
            continue
        del actual
        elapsed = _cold_graph_time(lambda: _run(x, weight, option), flush)
        if elapsed < best_ms * 0.985:
            best, best_ms = option, elapsed
    if best is not None:
        native_ms = _cold_graph_time(lambda: _run(x, weight, None), flush)
        best_ms = _cold_graph_time(lambda: _run(x, weight, best), flush)
        if best_ms >= native_ms * 0.985:
            best = None
    print(f"gated projection warmup: rows={x.shape[0]} backend={best or 'cublas+swiglu'} ratio={best_ms / native_ms:.3f}", flush=True)
    return best


def gated_linear(x, weight):
    """BF16 [...,K] and contiguous packed [2*I,K] -> SwiGLU(gate, up) [...,I]; prefill-sized inputs."""
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    key = (x.device, flat.shape[0], weight.shape[0], flat.shape[1])
    if key not in _CHOICES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("gated projection selection must finish during eager warmup")
        _CHOICES[key] = _choose(flat, weight)
    return _run(flat, weight, _CHOICES[key]).reshape(*x.shape[:-1], weight.shape[0] // 2)
