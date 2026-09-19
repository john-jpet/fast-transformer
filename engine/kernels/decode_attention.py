"""Dense GQA decode with a device-side length and split-KV softmax reduction.

BF16 Q/K/V, FP32 scores and accumulation, BF16 output. Each partial attends to
a disjoint interval; the second kernel combines their softmax normalizers.
Every valid cache position contributes. No per-token host synchronization.
"""

import statistics
import time

import torch
import triton
import triton.language as tl

from kernels.tune import register


@triton.jit
def _decode_partials(
    q_ptr, k_ptr, v_ptr, position_ptr, partial_ptr, stats_ptr,
    GROUPS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)  # flattened (batch, KV head)
    split = tl.program_id(1)
    heads = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, DIM)
    columns = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr + (group * GROUPS + heads[:, None]) * DIM + dims[None, :],
        heads[:, None] < GROUPS, other=0,
    )
    valid = tl.load(position_ptr).to(tl.int32) + 1
    begin = split * CHUNK
    end = tl.minimum(tl.minimum(begin + CHUNK, CAPACITY), valid)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    cache_base = group * CAPACITY * DIM
    # Whole tiles below ``end`` need no masks; only the last tile is ragged.
    whole = begin + tl.maximum(end - begin, 0) // BLOCK_N * BLOCK_N
    for start in range(begin, whole, BLOCK_N):
        tokens = start + columns
        key = tl.load(k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None])
        scores = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        value = tl.load(v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :])
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    for start in range(whole, end, BLOCK_N):
        tokens = start + columns
        key = tl.load(
            k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None],
            tokens[None, :] < end, other=0,
        )
        scores = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        scores = tl.where(tokens[None, :] < end, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        value = tl.load(
            v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
            tokens[:, None] < end, other=0,
        )
        # Standard BF16 flash-attention product with FP32 accumulation.
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    partial_base = ((group * SPLITS + split) * GROUPS + heads) * DIM
    tl.store(
        partial_ptr + partial_base[:, None] + dims[None, :], accumulator,
        heads[:, None] < GROUPS,
    )
    stats_base = ((group * SPLITS + split) * GROUPS + heads) * 2
    # An empty interval has m=-inf, l=0, accumulator=0. The merge gives it
    # exactly zero weight, including when valid length is shorter than CHUNK.
    tl.store(stats_ptr + stats_base, maximum, heads < GROUPS)
    tl.store(stats_ptr + stats_base + 1, denominator, heads < GROUPS)


@triton.jit
def _decode_merge(
    partial_ptr, stats_ptr, out_ptr,
    GROUPS: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    group = head // GROUPS
    within_group = head % GROUPS
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, DIM)
    offset = (group * SPLITS + splits) * GROUPS + within_group
    maxima = tl.load(stats_ptr + offset * 2, splits < SPLITS, other=-float("inf"))
    denominators = tl.load(stats_ptr + offset * 2 + 1, splits < SPLITS, other=0)
    maximum = tl.max(maxima, axis=0)
    correction = tl.exp2(maxima - maximum)
    partials = tl.load(
        partial_ptr + offset[:, None] * DIM + dims[None, :],
        splits[:, None] < SPLITS, other=0,
    )
    denominator = tl.sum(denominators * correction, axis=0)
    numerator = tl.sum(partials * correction[:, None], axis=0)
    tl.store(out_ptr + head * DIM + dims, numerator / denominator)


def _attend(query, key, value, position, scale, config):
    batch, query_heads, _, dim = query.shape
    kv_heads, capacity = key.shape[1:3]
    groups = query_heads // kv_heads
    block_n, splits, warps = config
    chunk = triton.cdiv(capacity, splits)
    partial = torch.empty((batch * kv_heads, splits, groups, dim), device=query.device, dtype=torch.float32)
    stats = torch.empty((batch * kv_heads, splits, groups, 2), device=query.device, dtype=torch.float32)
    out = torch.empty((batch, 1, query_heads, dim), device=query.device, dtype=query.dtype)
    _decode_partials[(batch * kv_heads, splits)](
        query, key, value, position, partial, stats,
        GROUPS=groups, DIM=dim, CAPACITY=capacity, SPLITS=splits, CHUNK=chunk,
        SCALE=scale, BLOCK_M=max(16, triton.next_power_of_2(groups)), BLOCK_N=block_n,
        num_warps=warps, num_stages=2,
    )
    _decode_merge[(batch * query_heads,)](
        partial, stats, out, GROUPS=groups, DIM=dim, SPLITS=splits,
        BLOCK_S=triton.next_power_of_2(splits), num_warps=4,
    )
    return out


def _default_config(batch, kv_heads, capacity):
    block_n = 32 if batch * kv_heads < 16 else 64
    # Enough independent KV intervals for small batches to occupy an H100.
    # Shape-only policy: neither token values nor sample number affect it.
    splits = min(32, triton.cdiv(256, batch * kv_heads), triton.cdiv(capacity, block_n))
    return (block_n, splits, 4)


def _graph_time(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            fn()
    graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


_CONFIGS = {}
_BLOCK_LAYOUTS = {}
_TUNING_SECONDS = 0.0  # the plain-decode layout search never found a winner; keep the default


def _choose(query, key, value, position, scale):
    """Measure a few shape-only split layouts once, before graph capture.

    Every layout is the same dense attention over every valid key; only the
    interval partition differs. Keep the official-run default unless another
    layout agrees with it and is measurably faster at this shape and length.
    """
    batch = query.shape[0]
    kv_heads, capacity = key.shape[1:3]
    default = _default_config(batch, kv_heads, capacity)
    block_n, splits, _ = default
    wanted = [
        (64, splits, 4), (128, max(1, splits // 2), 4), (64, max(1, splits // 2), 4),
        (64, min(32, splits * 2), 4), (128, splits, 4), (128, max(1, splits // 4), 4),
    ]
    candidates = []
    for config in wanted[:3]:
        config = (config[0], max(1, min(config[1], triton.cdiv(capacity, config[0]))), config[2])
        if config != default and config not in candidates:
            candidates.append(config)
    agreeing = []
    deadline = time.monotonic() + _TUNING_SECONDS
    reference = _attend(query, key, value, position, scale, default)
    best, best_ms = default, _graph_time(lambda: _attend(query, key, value, position, scale, default))
    for config in candidates:
        if time.monotonic() >= deadline:
            break
        actual = _attend(query, key, value, position, scale, config)
        if not torch.allclose(actual.float(), reference.float(), atol=0.02, rtol=0.02):
            continue
        agreeing.append(config)
        elapsed = _graph_time(lambda: _attend(query, key, value, position, scale, config))
        if elapsed < best_ms * 0.97:
            best, best_ms = config, elapsed
    if best != default:
        # Recheck both after all compilation, so clock ramp cannot bias it.
        default_ms = _graph_time(lambda: _attend(query, key, value, position, scale, default))
        best_ms = _graph_time(lambda: _attend(query, key, value, position, scale, best))
        if best_ms >= default_ms * 0.97:
            best = default
    print(f"decode attention warmup: layout={best} default={default}", flush=True)
    return best, [default, *agreeing]


def decode_attention(query, key, value, position, scale):
    """Q [B,Hq,1,D], KV [B,Hkv,C,D] -> [B,1,Hq,D]."""
    batch, query_heads, tokens, dim = query.shape
    kv_heads, capacity = key.shape[1:3]
    assert tokens == 1 and query_heads % kv_heads == 0
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert value.shape == key.shape and key.shape[0] == batch and key.shape[3] == dim
    assert dim in (64, 128) and position.shape == (1,) and position.dtype == torch.int64
    shape = (query.device, batch, query_heads, kv_heads, capacity, dim)
    if shape not in _CONFIGS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("attention layout selection must finish during eager warmup")
        _CONFIGS[shape], agreeing = _choose(query, key, value, position, scale)
        # Dense attention reads the whole KV prefix every step: weigh it highly.
        register(
            ("decode_attention",) + shape[1:], batch, 1 << 40, agreeing,
            lambda: _CONFIGS[shape], lambda config: _CONFIGS.__setitem__(shape, config),
        )
    return _attend(query, key, value, position, scale, _CONFIGS[shape])


@triton.jit
def _block_partials(
    q_ptr, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
    TOKENS: tl.constexpr, GROUPS: tl.constexpr, Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    DIM: tl.constexpr, CAPACITY: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Interval softmax for the TOKENS successive queries of one row and KV head.

    Query t of row b sits at position[b] + t and sees slots up to itself. All
    TOKENS * GROUPS queries share each loaded K/V tile; only their masks differ.
    Q is token-major [B,T,Hq,D]; same BF16/FP32 arithmetic as ``_decode_partials``.
    """
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
    query = tl.load(q_ptr + q_offset[:, None] + dims[None, :], live[:, None], other=0)
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
    # Tiles wholly inside the known prefix are visible to every query of the
    # block: no load masks, no visibility select. Same values, same order.
    whole = begin + tl.maximum(tl.minimum(end, first) - begin, 0) // BLOCK_N * BLOCK_N
    for start in range(begin, whole, BLOCK_N):
        tokens = start + columns
        key = tl.load(k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None])
        scores = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        value = tl.load(v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :])
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    for start in range(whole, end, BLOCK_N):
        tokens = start + columns
        key = tl.load(
            k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None],
            tokens[None, :] < end, other=0,
        )
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
            tokens[:, None] < end, other=0,
        )
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    if SPLITS == 1:
        # One interval covers the whole prefix: normalize and write the final
        # token-major output here; no partial storage, no merge launch.
        # (Single-pass path contributed by john-jpet's fork.)
        tl.store(
            out_ptr + q_offset[:, None] + dims[None, :],
            accumulator / denominator[:, None], live[:, None],
        )
    else:
        slot = (group * SPLITS + split) * (TOKENS * GROUPS) + members
        tl.store(partial_ptr + slot[:, None] * DIM + dims[None, :], accumulator, live[:, None])
        tl.store(stats_ptr + slot * 2, maximum, live)
        tl.store(stats_ptr + slot * 2 + 1, denominator, live)


@triton.jit
def _block_merge(
    partial_ptr, stats_ptr, out_ptr,
    TOKENS: tl.constexpr, GROUPS: tl.constexpr, Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    DIM: tl.constexpr, SPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64)  # flattened (row, token, query head) = output order
    head = index % Q_HEADS
    token = (index // Q_HEADS) % TOKENS
    row = index // (Q_HEADS * TOKENS)
    group = row * KV_HEADS + head // GROUPS
    member = token * GROUPS + head % GROUPS
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, DIM)
    slot = (group * SPLITS + splits) * (TOKENS * GROUPS) + member
    maxima = tl.load(stats_ptr + slot * 2, splits < SPLITS, other=-float("inf"))
    denominators = tl.load(stats_ptr + slot * 2 + 1, splits < SPLITS, other=0)
    maximum = tl.max(maxima, axis=0)
    correction = tl.exp2(maxima - maximum)
    partials = tl.load(partial_ptr + slot[:, None] * DIM + dims[None, :], splits[:, None] < SPLITS, other=0)
    denominator = tl.sum(denominators * correction, axis=0)
    numerator = tl.sum(partials * correction[:, None], axis=0)
    tl.store(out_ptr + index * DIM + dims, numerator / denominator)


def block_attention(query, key, value, position, scale, chain):
    """Verify blocks: token-major Q [B,T,Hq,D], KV [B,Hkv,C,D], position [B] -> [B,T,Hq,D].

    Token t of row b occupies slot ``position[b] + t``. The first ``chain[b]``
    tokens attend causally; the rest are alternatives to token 1 and attend to
    the prefix through token 0 plus themselves. The block's own keys must
    already be in the cache.
    """
    batch, tokens, query_heads, dim = query.shape
    kv_heads, capacity = key.shape[1:3]
    assert key.shape[0] == batch and query_heads % kv_heads == 0
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert value.shape == key.shape and key.shape[3] == dim and dim in (64, 128)
    assert position.shape == (batch,) and position.dtype == torch.int64
    assert chain.shape == (batch,) and chain.dtype == torch.int64
    groups = query_heads // kv_heads
    members = tokens * groups
    shape = (query.device, batch, tokens, query_heads, kv_heads, capacity, dim)
    if shape not in _BLOCK_LAYOUTS:
        # Start from the decode default; the captured verify graph re-judges
        # the alternatives (same dense attention, different interval tiling).
        default = _default_config(batch, kv_heads, capacity)
        options = [default]
        for block_n, splits in ((64, 1), (128, 1), (128, default[1] // 2), (64, default[1] * 2), (128, default[1])):
            option = (block_n, max(1, min(32, splits, triton.cdiv(capacity, block_n))), 4)
            if option not in options:
                options.append(option)
        _BLOCK_LAYOUTS[shape] = default
        if not torch.cuda.is_current_stream_capturing():
            register(
                ("block_attention",) + shape[1:], batch * tokens, 1 << 41, options,
                lambda: _BLOCK_LAYOUTS[shape], lambda option: _BLOCK_LAYOUTS.__setitem__(shape, option),
            )
    block_n, splits, warps = _BLOCK_LAYOUTS[shape]
    chunk = triton.cdiv(capacity, splits)
    out = torch.empty((batch, tokens, query_heads, dim), device=query.device, dtype=query.dtype)
    if splits == 1:
        partial = stats = out  # unused pointers in that specialization
    else:
        partial = torch.empty((batch * kv_heads, splits, members, dim), device=query.device, dtype=torch.float32)
        stats = torch.empty((batch * kv_heads, splits, members, 2), device=query.device, dtype=torch.float32)
    _block_partials[(batch * kv_heads, splits)](
        query, key, value, position, chain, partial, stats, out,
        TOKENS=tokens, GROUPS=groups, Q_HEADS=query_heads, KV_HEADS=kv_heads,
        DIM=dim, CAPACITY=capacity, SPLITS=splits, CHUNK=chunk, SCALE=scale,
        BLOCK_M=max(16, triton.next_power_of_2(members)), BLOCK_N=block_n,
        num_warps=warps, num_stages=2,
    )
    if splits > 1:
        _block_merge[(batch * tokens * query_heads,)](
            partial, stats, out,
            TOKENS=tokens, GROUPS=groups, Q_HEADS=query_heads, KV_HEADS=kv_heads,
            DIM=dim, SPLITS=splits, BLOCK_S=triton.next_power_of_2(splits), num_warps=4,
        )
    return out
