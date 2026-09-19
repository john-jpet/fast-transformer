"""Q/K head RMSNorm, RoPE, and in-place KV-cache update."""

import torch
import triton
import triton.language as tl

from kernels.tune import pick


@triton.jit
def _qk_rope_cache(
    packed, q_weight, k_weight, cos, sin, position, query, keys, values,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
    CAPACITY: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
    TOKENS: tl.constexpr, PREFILL: tl.constexpr, BLOCK: tl.constexpr,
    ROWS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // TOKENS
    token = row % TOKENS
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    paired = (col + DIM // 2) % DIM
    packed_row = row * (Q_HEADS + 2 * KV_HEADS) * DIM
    offset = packed_row + head * DIM
    x = tl.load(packed + offset + col, valid, other=0).to(tl.float32)
    x_pair = tl.load(packed + offset + paired, valid, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / DIM
    eps = tl.where(head < Q_HEADS, Q_EPS, K_EPS)
    inv_std = tl.rsqrt(variance + eps)
    if head < Q_HEADS:
        gain = tl.load(q_weight + col, valid, other=0).to(tl.float32)
        gain_pair = tl.load(q_weight + paired, valid, other=0).to(tl.float32)
    else:
        gain = tl.load(k_weight + col, valid, other=0).to(tl.float32)
        gain_pair = tl.load(k_weight + paired, valid, other=0).to(tl.float32)

    # Native norm: FP32 normalization -> BF16 -> learned gain -> BF16.
    normalized = (x * inv_std).to(tl.bfloat16).to(tl.float32)
    normalized_pair = (x_pair * inv_std).to(tl.bfloat16).to(tl.float32)
    weighted = (normalized * gain).to(tl.bfloat16).to(tl.float32)
    weighted_pair = (normalized_pair * gain_pair).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < DIM // 2, -weighted_pair, weighted_pair)
    # Phases are shared by the batch rows, or per row for verify blocks whose
    # rows sit at different positions.
    phase = token
    if ROWS:
        phase = row
    cosine = tl.load(cos + phase * DIM + col, valid, other=0).to(tl.float32)
    sine = tl.load(sin + phase * DIM + col, valid, other=0).to(tl.float32)
    # Native RoPE rounds BOTH products before their BF16 addition. Keeping
    # these products in FP32 and rounding only the sum changes the function.
    direct = (weighted * cosine).to(tl.bfloat16).to(tl.float32)
    turn = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = direct + turn
    if head < Q_HEADS:
        tl.store(query + (row * Q_HEADS + head) * DIM + col, result, valid)
    else:
        kv_head = head - Q_HEADS
        if PREFILL:
            pos = token
        else:
            # Token t of a decode block lands at position + t (t is 0 for the
            # ordinary single-token step).
            if ROWS:
                pos = tl.load(position + batch).to(tl.int64) + token
            else:
                pos = tl.load(position).to(tl.int64) + token
        cache_offset = ((batch * KV_HEADS + kv_head) * CAPACITY + pos) * DIM + col
        tl.store(keys + cache_offset, result, valid)
        value_offset = packed_row + (Q_HEADS + KV_HEADS + kv_head) * DIM + col
        value = tl.load(packed + value_offset, valid, other=0)
        tl.store(values + cache_offset, value, valid)


def qk_rope_cache(packed, q_norm, k_norm, cos, sin, position, keys, values, q_heads, prefill=False, rows=False):
    """BF16 packed QKV and [B,Hkv,C,D] caches, for prefill or one decode token.

    Native BF16 phases [1,T,D] are shared by batch rows. Prefill writes [0,T);
    decode writes T slots starting at the device-side scalar position. Return a [B,Hq,T,D]
    view of token-major Q storage (contiguous when T is one).
    """
    batch, kv_heads, capacity, dim = keys.shape
    tokens = packed.shape[1]
    assert packed.dtype == keys.dtype == values.dtype == torch.bfloat16
    assert packed.is_contiguous() and keys.is_contiguous() and values.is_contiguous()
    assert packed.shape == (batch, tokens, (q_heads + 2 * kv_heads) * dim)
    assert 0 < tokens <= capacity
    assert values.shape == keys.shape and dim % 2 == 0
    assert not (rows and prefill)
    assert cos.numel() == sin.numel() == (batch if rows else 1) * tokens * dim
    assert cos.is_contiguous() and sin.is_contiguous()
    assert position.shape == ((tokens,) if prefill else (batch,) if rows else (1,))
    assert position.dtype == torch.int64
    # Token-major storage: Flash then returns a token-major output, so the
    # caller's transpose back to [B,T,Hq,D] is already contiguous.
    query = torch.empty((batch, tokens, q_heads, dim), device=packed.device, dtype=packed.dtype)
    def launch(warps):
        _qk_rope_cache[(batch * tokens, q_heads + kv_heads)](
            packed, q_norm.weight, k_norm.weight, cos, sin, position, query, keys, values,
            Q_HEADS=q_heads, KV_HEADS=kv_heads, DIM=dim, CAPACITY=capacity,
            Q_EPS=q_norm.variance_epsilon, K_EPS=k_norm.variance_epsilon,
            TOKENS=tokens, PREFILL=prefill, BLOCK=triton.next_power_of_2(dim), ROWS=rows,
            num_warps=warps,
        )

    # Every option writes the same slots with the same values.
    launch(1 if prefill else pick(("qk_rope", batch * tokens, q_heads, kv_heads, dim, capacity, rows), 4, (1, 2), launch))
    return query.transpose(1, 2)
