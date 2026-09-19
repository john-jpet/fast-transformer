"""Build an unsubmitted architecture prototype from the validated attention loop."""
from pathlib import Path
root = Path(__file__).resolve().parents[2]
source = (root / 'engine/kernels/decode_attention.py').read_text()
body = source[source.index('@triton.jit\ndef _block_partials'):source.index('@triton.jit\ndef _block_merge')]
body = body.replace('def _block_partials(', 'def _fused_block(').replace(
    'q_ptr, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,',
    'packed, q_gain, k_gain, cos, sin, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,')
body = body.replace('DIM: tl.constexpr, CAPACITY:', 'Q_EPS: tl.constexpr, K_EPS: tl.constexpr, DIM: tl.constexpr, CAPACITY:')
body = body.replace('    query = tl.load(q_ptr + q_offset[:, None] + dims[None, :], live[:, None], other=0)', '''    packed_width = (Q_HEADS + 2 * KV_HEADS) * DIM
    q_source = (row * TOKENS + token) * packed_width + q_head * DIM
    query = _normalize_rope(packed, q_source, q_gain, cos, sin,
                           row * TOKENS + token, live, DIM, Q_EPS)''')
body = body.replace('tokens[None, :] < end, other=0,', 'tokens[None, :] < first - 1, other=0,')
body = body.replace('        scores = tl.dot(query, key)', '''        # Historical loads never observe writes from this invocation.
        # One program owns this row/KV head (SPLITS must be one).
        fresh = (tokens >= first - 1) & (tokens < end)
        current = tokens - (first - 1)
        packed_rows = row * TOKENS + current
        if start + BLOCK_N > first - 1:
            k_source = packed_rows * packed_width + (Q_HEADS + kv_head) * DIM
            new_key = _normalize_rope(packed, k_source, k_gain, cos, sin,
                                      packed_rows, fresh, DIM, K_EPS)
            key = tl.where(fresh[None, :], tl.trans(new_key), key)
            tl.store(k_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
                     new_key, fresh[:, None])
        scores = tl.dot(query, key)''')
body = body.replace('tokens[:, None] < end, other=0,', 'tokens[:, None] < first - 1, other=0,')
body = body.replace('        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)', '''        if start + BLOCK_N > first - 1:
            v_source = packed_rows * packed_width + (Q_HEADS + KV_HEADS + kv_head) * DIM
            new_value = tl.load(packed + v_source[:, None] + dims[None, :], fresh[:, None], other=0)
            value = tl.where(fresh[:, None], new_value, value)
            tl.store(v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
                     new_value, fresh[:, None])
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)''')
body = body.replace('    group = tl.program_id(0)', '    tl.static_assert(SPLITS == 1)\n    group = tl.program_id(0)', 1)
header = '''"""UNSUBMITTED prototype: QK norm/RoPE/cache writes fused into unsplit attention."""
import triton
import triton.language as tl

@triton.jit
def _normalize_rope(packed, offsets, gain, cos, sin, phases, live,
                    DIM: tl.constexpr, EPS: tl.constexpr):
    dims = tl.arange(0, DIM)
    paired = (dims + DIM // 2) % DIM
    x = tl.load(packed + offsets[:, None] + dims[None, :], live[:, None], other=0).to(tl.float32)
    xp = tl.load(packed + offsets[:, None] + paired[None, :], live[:, None], other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x*x, axis=1) / DIM + EPS)
    w = tl.load(gain + dims).to(tl.float32)
    wp = tl.load(gain + paired).to(tl.float32)
    y = ((x * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w[None, :]).to(tl.bfloat16).to(tl.float32)
    yp = ((xp * inv[:, None]).to(tl.bfloat16).to(tl.float32) * wp[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(cos + phases[:, None]*DIM + dims[None, :], live[:, None], other=0).to(tl.float32)
    s = tl.load(sin + phases[:, None]*DIM + dims[None, :], live[:, None], other=0).to(tl.float32)
    rotated = tl.where(dims[None, :] < DIM//2, -yp, yp)
    direct = (y*c).to(tl.bfloat16).to(tl.float32)
    turn = (rotated*s).to(tl.bfloat16).to(tl.float32)
    return (direct+turn).to(tl.bfloat16)

'''
(root / 'agent/lab/fused_attention_prototype.py').write_text(header + body)
print('Built unsubmitted fused attention prototype')
