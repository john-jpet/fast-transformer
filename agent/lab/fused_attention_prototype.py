"""UNSUBMITTED prototype: QK norm/RoPE/cache writes fused into unsplit attention."""
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

@triton.jit
def _fused_block(
    packed, q_gain, k_gain, cos, sin, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
    TOKENS: tl.constexpr, GROUPS: tl.constexpr, Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    Q_EPS: tl.constexpr, K_EPS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Interval softmax for the TOKENS successive queries of one row and KV head.

    Query t of row b sits at position[b] + t and sees slots up to itself. All
    TOKENS * GROUPS queries share each loaded K/V tile; only their masks differ.
    Q is token-major [B,T,Hq,D]; same BF16/FP32 arithmetic as ``_decode_partials``.
    """
    tl.static_assert(SPLITS == 1)
    group = tl.program_id(0).to(tl.int64)  # flattened (row, KV head)
    split = tl.program_id(1)
    row = group // KV_HEADS
    kv_head = group % KV_HEADS
    members = tl.arange(0, BLOCK_M)  # flattened (token, query head within the KV group)
    live = members < TOKENS * GROUPS
    token = members // GROUPS
    dims = tl.arange(0, DIM)
    columns = tl.arange(0, BLOCK_N)
    q_head = kv_head * GROUPS + members % GROUPS
    q_offset = ((row * TOKENS + token) * Q_HEADS + q_head) * DIM
    packed_width = (Q_HEADS + 2 * KV_HEADS) * DIM
    q_source = (row * TOKENS + token) * packed_width + q_head * DIM
    query = _normalize_rope(packed, q_source, q_gain, cos, sin,
                           row * TOKENS + token, live, DIM, Q_EPS)
    first = tl.load(position_ptr + row).to(tl.int32) + 1
    # Tokens below CHAIN form a causal chain. Later tokens are alternative
    # first drafts: each sees the prefix through the trusted token and its own
    # slot only, never the chain or another alternative.
    chained = token < tl.load(chain_ptr + row)
    valid = first + tl.where(chained, token, 0).to(tl.int32)
    own = tl.where(chained, -1, first - 1 + token.to(tl.int32))
    begin = split * CHUNK
    end = tl.minimum(tl.minimum(begin + CHUNK, CAPACITY), first + (TOKENS - 1))
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    cache_base = group * CAPACITY * DIM
    for start in range(begin, end, BLOCK_N):
        tokens = start + columns
        key = tl.load(
            k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None],
            tokens[None, :] < first - 1, other=0,
        )
        # Historical loads never observe writes from this invocation.
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
        scores = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        visible = (tokens[None, :] < end) & ((tokens[None, :] < valid[:, None]) | (tokens[None, :] == own[:, None]))
        scores = tl.where(visible, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # A query that has seen nothing yet keeps -inf; never form -inf - -inf.
        pivot = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = tl.exp2(scores - pivot[:, None])
        correction = tl.exp2(maximum - pivot)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        value = tl.load(
            v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
            tokens[:, None] < first - 1, other=0,
        )
        if start + BLOCK_N > first - 1:
            v_source = packed_rows * packed_width + (Q_HEADS + KV_HEADS + kv_head) * DIM
            new_value = tl.load(packed + v_source[:, None] + dims[None, :], fresh[:, None], other=0)
            value = tl.where(fresh[:, None], new_value, value)
            tl.store(v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
                     new_value, fresh[:, None])
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    if SPLITS == 1:
        # The whole prefix was reduced in this program. Write token-major
        # final values directly, without partial storage or a merge launch.
        tl.store(out_ptr + q_offset[:, None] + dims[None, :],
                 accumulator / denominator[:, None], live[:, None])
    else:
        slot = (group * SPLITS + split) * (TOKENS * GROUPS) + members
        tl.store(partial_ptr + slot[:, None] * DIM + dims[None, :], accumulator, live[:, None])
        tl.store(stats_ptr + slot * 2, maximum, live)
        tl.store(stats_ptr + slot * 2 + 1, denominator, live)


