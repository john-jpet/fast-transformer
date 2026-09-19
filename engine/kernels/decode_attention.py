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

from kernels.pdl import wait as pdl_wait

from kernels.tune import register


def _wide_prefix(batch, capacity):
    """Whether the unmasked whole-prefix loop pays for its extra code.

    Measured on the platform (public TPOT, same drafts): with it, batch 16 x
    640 slots got ~2.5% faster and batch 4 x 2086 stayed level, but batch 1 x
    560 lost ~3-6% of its pass: there the kernel is bound by how many programs
    fit on the device, not by the bytes of K and V it reads.
    """
    return batch * capacity >= 6000


@triton.jit
def _decode_partials(
    q_ptr, k_ptr, v_ptr, position_ptr, partial_ptr, stats_ptr,
    GROUPS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREFIX: tl.constexpr = False,
):
    pdl_wait()  # before any global memory access
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
    # PREFIX off: the first loop is empty and compiles away (see _wide_prefix).
    whole = begin
    if PREFIX:
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
    pdl_wait()  # before any global memory access
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
        PREFIX=_wide_prefix(batch, capacity), num_warps=warps, num_stages=2,
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
#: Batches above 16 decode one token per step with no speculation, and at long
#: context the K/V read is most of that step: those shapes get a real search.
_WIDE_TUNING_SECONDS = 4.0


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
    budget = _TUNING_SECONDS if query.shape[0] <= 16 else _WIDE_TUNING_SECONDS
    deadline = time.monotonic() + budget
    reference = _attend(query, key, value, position, scale, default)
    # No search budget, no baseline timing: that graph capture would be wasted.
    best, best_ms = default, (
        _graph_time(lambda: _attend(query, key, value, position, scale, default)) if budget > 0 else 0.0
    )
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
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PREFIX: tl.constexpr = False,
):
    """Interval softmax for the TOKENS successive queries of one row and KV head.

    Query t of row b sits at position[b] + t and sees slots up to itself. All
    TOKENS * GROUPS queries share each loaded K/V tile; only their masks differ.
    Q is token-major [B,T,Hq,D]; same BF16/FP32 arithmetic as ``_decode_partials``.
    """
    pdl_wait()  # before any global memory access
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
    whole = begin
    if PREFIX:
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
def _block_partials_tma(
    q_ptr, k_ptr, v_ptr, kd_ptr, vd_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
    TOKENS: tl.constexpr, GROUPS: tl.constexpr, Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    DIM: tl.constexpr, CAPACITY: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LIMIT: tl.constexpr,
    TMA: tl.constexpr = False,
):
    """``_block_partials`` (PREFIX on) with the whole-prefix K/V tiles read through Hopper TMA tensor maps.

    ``kd_ptr``/``vd_ptr`` are device descriptors of this layer's K and V viewed
    as [B*Hkv*C rows, DIM], tiled [BLOCK_N, DIM]; a load takes the ELEMENT
    offsets (row, 0) of a whole tile and returns it in storage orientation
    [BLOCK_N, DIM]. Transposing a descriptor-loaded tile into ``tl.dot`` segfaults
    the Triton 3.1.0 compiler, so the prefix loop turns the PRODUCT instead:
    ``tl.dot(key, query^T)`` -> [BLOCK_N, BLOCK_M], softmax statistics along
    axis 0, and only the COMPUTED probabilities are transposed for the V product.
    Same tiles, same order, same BF16 operands and FP32 accumulation as
    ``_block_partials``. TMA off: the same source with ordinary loads - the
    fail-safe twin, launched with the same grid and constants. K and V loads
    are adjacent (one barrier) and every carry is redefined in the body, which
    is what lets the 3.1 pipeliner prefetch the next tiles behind the dots.
    Only tiles below LIMIT (rows the descriptor covers) and wholly inside the
    known prefix are read this way; the ragged tail is ``_block_partials``'s.
    """
    pdl_wait()  # before any global memory access
    pid = tl.program_id(0)
    group = pid.to(tl.int64)  # flattened (row, KV head)
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
    query_t = tl.load(q_ptr + q_offset[None, :] + dims[:, None], live[None, :], other=0)
    first = tl.load(position_ptr + row).to(tl.int32) + 1
    chained = token < tl.load(chain_ptr + row)
    valid = first + tl.where(chained, token, 0).to(tl.int32)
    own = tl.where(chained, -1, first - 1 + token.to(tl.int32))
    begin = split * CHUNK
    end = tl.minimum(tl.minimum(begin + CHUNK, CAPACITY), first + (TOKENS - 1))
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    cache_base = group * CAPACITY * DIM
    cache_row = pid * CAPACITY  # int32: the launcher bounds B * Hkv * C
    inside = tl.minimum(tl.minimum(end, first), LIMIT)
    whole = begin + tl.maximum(inside - begin, 0) // BLOCK_N * BLOCK_N
    for start in range(begin, whole, BLOCK_N):
        if TMA:
            key = tl._experimental_descriptor_load(kd_ptr, [cache_row + start, 0], [BLOCK_N, DIM], tl.bfloat16)
            value = tl._experimental_descriptor_load(vd_ptr, [cache_row + start, 0], [BLOCK_N, DIM], tl.bfloat16)
        else:
            tokens = start + columns
            key = tl.load(k_ptr + cache_base + tokens[:, None] * DIM + dims[None, :])
            value = tl.load(v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :])
        scores = tl.dot(key, query_t) * (SCALE * 1.4426950408889634)  # [BLOCK_N, BLOCK_M]
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        probabilities = tl.exp2(scores - next_maximum[None, :])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(tl.trans(probabilities).to(tl.bfloat16), value, accumulator)
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
    pdl_wait()  # before any global memory access
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


# --- Hopper TMA reads of the whole-prefix K/V tiles ----------------------------
# A block layout (block_n, splits, warps, TMA, stages) is the same interval tiling
# with the prefix tiles read through tensor maps (``_block_partials_tma``) in a
# loop pipelined ``stages`` deep. It is offered only where the descriptor API
# exists and the unmasked prefix loop is in use, is never the default (the
# captured verify graph must time it faster), must first reproduce ordinary
# loads bit for bit on a private random cache (``_tma_probe``), and fails closed:
# no descriptor (capture before an eager pass, odd shape, retired) means the
# ordinary ``_block_partials`` launch of the same layout.
TMA = "tma"
_TMA_ATTENTION_OFF = [False]
_TMA_CHECKED = {}


def _tma_attention_retire(error):
    if not _TMA_ATTENTION_OFF[0]:
        print(f"TMA attention retired: {error!r}", flush=True)
    _TMA_ATTENTION_OFF[0] = True


def _tma_attention_offered(key):
    try:
        from kernels import linear
        return not _TMA_ATTENTION_OFF[0] and key.is_cuda and key.shape[3] == 128 and linear._tma_fill() is not None
    except Exception as error:
        _tma_attention_retire(error)
        return False


def _tma_maps(key, value, block_n):
    """(K descriptor, V descriptor, LIMIT) of one layer's cache, or None.

    The cache is described as a 2-D array [B*Hkv*C rows, D] tiled [block_n, D],
    with the row count rounded DOWN to whole tiles (driver.c asserts on a
    rejected encoding, so only whole 128-byte-aligned tilings are submitted, as
    for the GEMM weights). LIMIT keeps every TMA tile of every (row, KV head)
    below the described rows; slots past it take the ordinary masked loop.
    Built eagerly, once per cache tensor, and kept alive by kernels.linear; the
    cache is one static allocation per DecodeState, so addresses never change
    under a captured graph.
    """
    try:
        if _TMA_ATTENTION_OFF[0]:
            return None
        from kernels import linear
        batch, kv_heads, capacity, dim = key.shape
        rows = batch * kv_heads * capacity
        covered = rows // block_n * block_n
        limit = capacity - (rows - covered)
        if dim != 128 or limit < block_n or rows * dim >= 2 ** 31 or value.shape != key.shape:
            return None
        maps = []
        for cache in (key, value):
            desc = linear._tma_descriptor(cache.view(rows, dim)[:covered], block_n, dim)
            if desc is None:
                return None
            maps.append(desc)
        return maps[0], maps[1], limit
    except Exception as error:
        _tma_attention_retire(error)
        return None


def _launch_block(kind, query, key, value, position, chain, scale, layout, maps=None):
    """One block-attention launch. kind: "plain" (``_block_partials``), "tma", or "twin" (its ordinary-load twin)."""
    batch, tokens, query_heads, dim = query.shape
    kv_heads, capacity = key.shape[1:3]
    groups = query_heads // kv_heads
    members = tokens * groups
    block_n, splits, warps = layout[:3]
    chunk = triton.cdiv(capacity, splits)
    out = torch.empty((batch, tokens, query_heads, dim), device=query.device, dtype=query.dtype)
    if splits == 1:
        partial = stats = out  # unused pointers in that specialization
    else:
        partial = torch.empty((batch * kv_heads, splits, members, dim), device=query.device, dtype=torch.float32)
        stats = torch.empty((batch * kv_heads, splits, members, 2), device=query.device, dtype=torch.float32)
    constants = dict(
        TOKENS=tokens, GROUPS=groups, Q_HEADS=query_heads, KV_HEADS=kv_heads,
        DIM=dim, CAPACITY=capacity, SPLITS=splits, CHUNK=chunk, SCALE=scale,
        BLOCK_M=max(16, triton.next_power_of_2(members)), BLOCK_N=block_n,
    )
    if kind == "plain":
        _block_partials[(batch * kv_heads, splits)](
            query, key, value, position, chain, partial, stats, out,
            PREFIX=_wide_prefix(batch, capacity), num_warps=warps, num_stages=2, **constants,
        )
    else:
        k_map, v_map, limit = maps if kind == "tma" else (key, value, capacity - batch * kv_heads * capacity % block_n)
        _block_partials_tma[(batch * kv_heads, splits)](
            query, key, value, k_map, v_map, position, chain, partial, stats, out,
            LIMIT=limit, TMA=kind == "tma", num_warps=warps, num_stages=(tuple(layout[4:]) or (2,))[0], **constants,
        )
    if splits > 1:
        _block_merge[(batch * tokens * query_heads,)](
            partial, stats, out,
            TOKENS=tokens, GROUPS=groups, Q_HEADS=query_heads, KV_HEADS=kv_heads,
            DIM=dim, SPLITS=splits, BLOCK_S=triton.next_power_of_2(splits), num_warps=4,
        )
    return out


def _tma_probe(query, key, scale, layout):
    """Whether tensor-map reads reproduce ordinary loads bit for bit at this shape and layout.

    Eager, once per (shape, layout), on a private random cache of the real
    shape (the real one may still hold the synthetic zero prefix, which proves
    nothing): rows ending at the last legal slot, in the middle and near the
    start, so every tile offset and LIMIT are exercised. Equal to the plain
    kernel is the target; equal to the ordinary-load twin only (the turned
    product rounding differently on the device: a reordering) still proves
    the reads. Anything else keeps TMA off for this layout.
    """
    batch, tokens = query.shape[:2]
    capacity = key.shape[2]
    if key.numel() * key.element_size() > 1 << 28:
        return False  # the private K and V would cost real peak memory: leave such shapes on ordinary loads
    generator = torch.Generator(device=key.device).manual_seed(1729)
    q, k, v = (
        torch.randn(like.shape, device=key.device, dtype=key.dtype, generator=generator) for like in (query, key, key)
    )
    position = torch.randint(0, capacity - tokens + 1, (batch,), device=key.device, generator=generator)
    position[0] = capacity - tokens
    position[batch // 2] = capacity // 2
    chain = torch.full((batch,), tokens, device=key.device, dtype=torch.int64)
    maps = _tma_maps(k, v, layout[0])
    if maps is None:
        return False
    out = _launch_block("tma", q, k, v, position, chain, scale, layout, maps)
    exact = same = torch.equal(out, _launch_block("plain", q, k, v, position, chain, scale, layout))  # compiled already
    if not same:
        same = torch.equal(out, _launch_block("twin", q, k, v, position, chain, scale, layout))
    print(f"TMA attention probe {layout}: equals plain={exact} equals ordinary loads={same}", flush=True)
    return same and bool(torch.isfinite(out.float()).all())


def _tma_block(query, key, value, position, chain, scale, layout, shape):
    """The TMA launch of ``layout``, or None when the caller must launch the ordinary kernel."""
    if _TMA_ATTENTION_OFF[0] or not _wide_prefix(query.shape[0], key.shape[2]):
        return None
    try:
        check = (shape, layout)
        if check not in _TMA_CHECKED:
            if torch.cuda.is_current_stream_capturing():
                return None
            _TMA_CHECKED[check] = False  # a probe that raises stays failed
            _TMA_CHECKED[check] = _tma_probe(query, key, scale, layout)
        maps = _tma_maps(key, value, layout[0]) if _TMA_CHECKED[check] else None  # inside a capture: a lookup only
        if maps is None:
            return None
        return _launch_block("tma", query, key, value, position, chain, scale, layout, maps)
    except Exception as error:
        _tma_attention_retire(error)
        return None


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
        if _wide_prefix(batch, capacity) and _tma_attention_offered(key):
            # Last, never the default: tensor-map reads of the prefix tiles must win a timing.
            # Pipeline depth 2 issues the next K/V copy behind this tile's dots; 3 keeps one more tile in flight.
            options += [default + (TMA, 2), default + (TMA, 3)]
        _BLOCK_LAYOUTS[shape] = default
        if not torch.cuda.is_current_stream_capturing():
            # Refined first, as in the best measured engine (candidate 57): the
            # traffic-ordered variant (candidates 66/67) coincided with a slower
            # batch-one pass.
            register(
                ("block_attention",) + shape[1:], batch * tokens, 1 << 41, options,
                lambda: _BLOCK_LAYOUTS[shape], lambda option: _BLOCK_LAYOUTS.__setitem__(shape, option),
            )
    layout = _BLOCK_LAYOUTS[shape]
    out = _tma_block(query, key, value, position, chain, scale, layout, shape) if TMA in layout[3:] else None
    if out is None:
        out = _launch_block("plain", query, key, value, position, chain, scale, layout)
    return out
