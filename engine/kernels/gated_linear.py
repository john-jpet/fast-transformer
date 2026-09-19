"""Gate/up projection with the native BF16 SwiGLU rounding boundaries.

The paired GEMM writes only the activated product. A split-K alternative
reuses the existing projection and fuses its FP32 merge with activation.
Warmup compares both with the existing linear + SwiGLU path; capture never
compiles, times, or changes a choice. Large prefill and skinny decode choose
independently. All weights and persistent activations remain BF16.
"""

import time

import torch
import triton
import triton.language as tl

from kernels.linear import _CHOICES as LINEAR_CHOICES
from kernels.linear import _cold_graph_time, _skinny_gemm, linear
from kernels.swiglu import swiglu
from kernels.tune import register


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
    # Group adjacent row tiles to reuse weight tiles through L2 in prefill.
    pid = tl.program_id(0)
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
    result = _activate(gate, up)
    tl.store(out_ptr + rows[:, None] * I + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < I))


@triton.jit
def _merge_activate(
    partial_ptr, out_ptr, M: tl.constexpr, I: tl.constexpr,
    SPLITS: tl.constexpr, BS: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // I
    column = offsets % I
    splits = tl.arange(0, BS)
    address = splits[:, None] * M * (2 * I) + row[None, :] * (2 * I) + column[None, :]
    mask = (splits[:, None] < SPLITS) & (offsets[None, :] < M * I)
    gate = tl.sum(tl.load(partial_ptr + address, mask, other=0), axis=0)
    up = tl.sum(tl.load(partial_ptr + address + I, mask, other=0), axis=0)
    tl.store(out_ptr + offsets, _activate(gate, up), offsets < M * I)


def _run(x, weight, option):
    """Contiguous BF16 x[M,K], packed weights[2*I,K] -> new BF16 [M,I]."""
    if option is None:
        return swiglu(linear(x, weight))
    m, k = x.shape
    width = weight.shape[0] // 2
    out = torch.empty((m, width), device=x.device, dtype=x.dtype)
    if option[0] == "paired":
        _, bm, bn, bk, warps = option
        _paired_projection[(triton.cdiv(m, bm) * triton.cdiv(width, bn),)](
            x, weight, out, M=m, I=width, K=k, BM=bm, BN=bn, BK=bk,
            num_warps=warps, num_stages=3,
        )
    else:
        _, bn, bk, splits, warps = option
        partial = torch.empty((splits, m, 2 * width), device=x.device, dtype=torch.float32)
        chunk = triton.cdiv(k, splits * bk) * bk
        _skinny_gemm[(triton.cdiv(2 * width, bn), splits)](
            x, weight, partial, M=m, N=2 * width, K=k, SPLITS=splits,
            CHUNK=chunk, BLOCK_N=bn, BLOCK_K=bk,
            BLOCK_M=16 if m <= 16 else 32, num_warps=warps, num_stages=2,
        )
        _merge_activate[(triton.cdiv(m * width, 512),)](
            partial, out, M=m, I=width, SPLITS=splits,
            BS=triton.next_power_of_2(splits), BLOCK=512, num_warps=4,
        )
    return out


_CHOICES = {}
_DEADLINES = {}


def _choose(x, weight):
    m, k = x.shape
    width = weight.shape[0] // 2
    phase = "decode" if m <= 32 else "prefill"
    now = time.monotonic()
    deadline = _DEADLINES.setdefault(phase, now + 12.0)
    if now >= deadline:
        return None, [None]
    _run(x, weight, None)  # Also initializes the fallback's tuners.
    if m <= 32:
        candidates = [("paired", 16 if m <= 16 else 32, 32, 64, 4)]
        selected = LINEAR_CHOICES.get((x.device, m, 2 * width, k))
        if selected is not None and selected[0] == "gemm" and selected[3] > 1:
            candidates.insert(0, ("merge", *selected[1:]))
    else:
        candidates = [("paired", 32, 64, 32, 4), ("paired", 64, 64, 32, 8)]
    # Operator checks use nonzero values, never the zero capture-warmup input.
    generator = torch.Generator(device=x.device).manual_seed(31415)
    probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    reference = _run(probe, weight, None)
    flush = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.int32)
    best, options = None, [None]
    native_ms = _cold_graph_time(lambda: _run(x, weight, None), flush)
    best_ms = native_ms
    for option in candidates:
        if time.monotonic() >= deadline:
            break
        actual = _run(probe, weight, option)
        # This is a gross-error screen; the teacher-forced judge still decides
        # full-model correctness. The SiLU formula and every BF16 cast stay put.
        if not torch.allclose(actual.float(), reference.float(), rtol=0.035, atol=0.02):
            continue
        options.append(option)
        elapsed = _cold_graph_time(lambda: _run(x, weight, option), flush)
        if elapsed < best_ms * 0.985:
            best, best_ms = option, elapsed
    if best is not None:
        native_ms = _cold_graph_time(lambda: _run(x, weight, None), flush)
        best_ms = _cold_graph_time(lambda: _run(x, weight, best), flush)
        if best_ms >= native_ms * 0.985:
            best = None
    print(f"gated projection warmup: rows={m} backend={best or 'existing'} ratio={best_ms / native_ms:.3f}", flush=True)
    return best, options


def gated_linear(x, weight):
    """BF16 [...,K], contiguous packed [2*I,K]; no mutation or output aliasing."""
    if x.dtype != torch.bfloat16 or not weight.is_contiguous():
        return swiglu(linear(x, weight))
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = flat.shape
    key = (x.device, m, weight.shape[0], k)
    if key not in _CHOICES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("gated projection selection must finish during eager warmup")
        _CHOICES[key], options = _choose(flat, weight)
        if m <= 32:
            register(
                ("gated_projection", *key[1:]), m, weight.numel() + 1, options,
                lambda: _CHOICES[key], lambda option: _CHOICES.__setitem__(key, option),
            )
    return _run(flat, weight, _CHOICES[key]).reshape(*x.shape[:-1], weight.shape[0] // 2)
