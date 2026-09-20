"""Pure-PyTorch references for the engine's Triton kernels, keyed by kernel name.

Each mirrors its kernel's semantics: the same flat-pointer addressing (tensors
are read as raw contiguous memory from their first element, as a kernel would),
the same FP32 accumulation and the same BF16 cast placement. Reduction ORDER is
not reproduced. The speculation kernels are emulated per row, line by line.
Every write is bounds-checked: an out-of-range slot is silent corruption on the
GPU and an AssertionError here.
"""

import math

import torch
from triton import reference

BF16, F32 = torch.bfloat16, torch.float32
LOG2E = 1.4426950408889634


def flat(tensor, count=None):
    """The memory a kernel sees through this tensor's pointer, as a 1-D view."""
    storage = tensor.untyped_storage().nbytes() // tensor.element_size() - tensor.storage_offset()
    view = torch.as_strided(tensor, (storage,), (1,))
    if count is not None:
        assert 0 <= count <= storage, f"kernel would address {count} elements of a {storage}-element allocation"
        view = view[:count]
    return view


def merged(ptr, count, COUNT, SPLITS):
    """kernels/merged.py::load_merged over offsets [0, count): BF16 values."""
    if SPLITS == 1:
        return flat(ptr, count)
    assert ptr.dtype == F32 and count <= COUNT
    data = flat(ptr, SPLITS * COUNT).view(SPLITS, COUNT)[:, :count]
    total = data[0].clone()
    for split in range(1, SPLITS):
        total += data[split]
    return total.to(BF16)


@reference("_rms_norm_kernel")
def _rms_norm(grid, x_ptr, w_ptr, y_ptr, row_stride, n_cols, eps, BLOCK, **launch):
    rows = grid[0]
    assert BLOCK >= n_cols
    x = flat(x_ptr, rows * row_stride).view(rows, row_stride)[:, :n_cols].to(F32)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = flat(y_ptr, rows * row_stride).view(rows, row_stride)[:, :n_cols]
    out.copy_(normed.to(y_ptr.dtype) * flat(w_ptr, n_cols))


@reference("_add_rms_norm_kernel")
def _add_rms_norm(grid, x_ptr, residual_ptr, w_ptr, out_ptr, sum_ptr, WIDTH, EPS, BLOCK, COUNT=1, SPLITS=1, **launch):
    rows = grid[0]
    assert BLOCK >= WIDTH
    n = rows * WIDTH
    x = merged(x_ptr, n, COUNT, SPLITS).view(rows, WIDTH).to(F32)
    residual = flat(residual_ptr, n).view(rows, WIDTH).to(F32)
    summed = (x + residual).to(BF16)
    summed32 = summed.to(F32)
    variance = summed32.pow(2).mean(-1, keepdim=True)
    normalized = (summed32 * torch.rsqrt(variance + EPS)).to(BF16).to(F32)
    flat(sum_ptr, n).view(rows, WIDTH).copy_(summed)
    flat(out_ptr, n).view(rows, WIDTH).copy_(normalized * flat(w_ptr, WIDTH).to(F32))


@reference("_block_argmax")
def _block_argmax(grid, logits, best_value, best_index, VOCAB, BLOCKS, BLOCK, **launch):
    assert grid[1] == BLOCKS and BLOCKS * BLOCK >= VOCAB > (BLOCKS - 1) * BLOCK
    rows = grid[0]
    values = flat(logits, rows * VOCAB).view(rows, VOCAB).to(F32)
    for block in range(BLOCKS):
        tile = values[:, block * BLOCK:min((block + 1) * BLOCK, VOCAB)]
        index = tile.argmax(-1)  # first maximum, as tl.max(..., tie_break_left)
        flat(best_value, rows * BLOCKS).view(rows, BLOCKS)[:, block] = tile.gather(1, index[:, None])[:, 0]
        flat(best_index, rows * BLOCKS).view(rows, BLOCKS)[:, block] = block * BLOCK + index


@reference("_first_best")
def _first_best(grid, best_value, best_index, out, BLOCKS, BLOCK_B, **launch):
    assert BLOCK_B >= BLOCKS
    rows = grid[0]
    chosen = flat(best_value, rows * BLOCKS).view(rows, BLOCKS).argmax(-1)
    flat(out, rows).copy_(flat(best_index, rows * BLOCKS).view(rows, BLOCKS).gather(1, chosen[:, None])[:, 0])


@reference("_fused_block_argmax")
def _fused_block_argmax(grid, x_ptr, weight_ptr, best_value, best_index, M, N, K, BLOCK_N, BLOCK_K, BLOCK_M, EVEN_M, **launch):
    assert grid[0] * BLOCK_N == N and K % BLOCK_K == 0 and BLOCK_M >= M
    x = flat(x_ptr, M * K).view(M, K).to(F32)
    weight = flat(weight_ptr, N * K).view(N, K).to(F32)
    logits = (x @ weight.T).to(BF16).to(F32)                       # rounded like the projection
    tiles = grid[0]
    values = flat(best_value, M * tiles).view(M, tiles)
    indices = flat(best_index, M * tiles).view(M, tiles)
    for tile in range(tiles):
        chunk = logits[:, tile * BLOCK_N:(tile + 1) * BLOCK_N]
        index = chunk.argmax(-1)                                    # first maximum
        values[:, tile] = chunk.gather(1, index[:, None])[:, 0]
        indices[:, tile] = tile * BLOCK_N + index


@reference("_embed_rms_norm_kernel")
def _embed_rms_norm_kernel(grid, ids_ptr, table_ptr, w_ptr, y_ptr, h_ptr, n_cols, eps, BLOCK, **launch):
    rows = grid[0]
    ids = flat(ids_ptr, rows).to(torch.int64)
    table = flat(table_ptr, (int(ids.max()) + 1) * n_cols).view(-1, n_cols)
    raw = table[ids]
    flat(h_ptr, rows * n_cols).view(rows, n_cols).copy_(raw)
    x = raw.to(F32)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    weight = flat(w_ptr, n_cols)
    flat(y_ptr, rows * n_cols).view(rows, n_cols).copy_(normed.to(BF16) * weight)


@reference("_swiglu")
def _swiglu(grid, packed, output, WIDTH, BLOCK, COUNT=1, SPLITS=1, **launch):
    rows = grid[0]
    assert grid[1] * BLOCK >= WIDTH
    data = merged(packed, rows * 2 * WIDTH, COUNT, SPLITS).view(rows, 2 * WIDTH).to(F32)
    gate, up = data[:, :WIDTH], data[:, WIDTH:]
    activated = (gate / (1.0 + torch.exp(-gate))).to(output.dtype).to(F32)
    flat(output, rows * WIDTH).view(rows, WIDTH).copy_(activated * up)


@reference("_qk_rope_cache")
def _qk_rope_cache(
    grid, packed, q_weight, k_weight, cos, sin, position, query, keys, values,
    Q_HEADS, KV_HEADS, DIM, CAPACITY, Q_EPS, K_EPS, TOKENS, PREFILL, BLOCK,
    ROWS=False, COUNT=1, SPLITS=1, phases=None, TABLE=False, **launch,
):
    rows = grid[0]
    assert grid[1] == Q_HEADS + KV_HEADS and BLOCK >= DIM and rows % TOKENS == 0
    batch = rows // TOKENS
    assert not TABLE or (ROWS and phases is not None)
    heads = Q_HEADS + 2 * KV_HEADS
    data = merged(packed, rows * heads * DIM, COUNT, SPLITS).view(rows, heads, DIM)
    x = data[:, :Q_HEADS + KV_HEADS].to(F32)
    eps = torch.cat((torch.full((Q_HEADS,), Q_EPS), torch.full((KV_HEADS,), K_EPS)))[None, :, None]
    inv_std = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    gain = torch.cat((
        flat(q_weight, DIM).to(F32).expand(Q_HEADS, DIM), flat(k_weight, DIM).to(F32).expand(KV_HEADS, DIM),
    ))[None]
    normalized = (x * inv_std).to(BF16).to(F32)
    weighted = (normalized * gain).to(BF16).to(F32)
    half = DIM // 2
    rotated = torch.cat((-weighted[..., half:], weighted[..., :half]), dim=-1)
    if TABLE:
        # Whole tables, indexed at position[b] + phases[b, t] per row.
        where = flat(position, batch).to(torch.int64)
        offsets = where.repeat_interleave(TOKENS) + flat(phases, rows).to(torch.int64)
        table = CAPACITY + 64  # the tables cover at least the capacity; read only what is indexed
        cosine = flat(cos, int(offsets.max()) * DIM + DIM).view(-1, DIM).to(F32)[offsets]
        sine = flat(sin, int(offsets.max()) * DIM + DIM).view(-1, DIM).to(F32)[offsets]
    else:
        count = rows if ROWS else TOKENS
        cosine = flat(cos, count * DIM).view(count, DIM).to(F32)
        sine = flat(sin, count * DIM).view(count, DIM).to(F32)
        if not ROWS:
            cosine, sine = cosine.repeat(batch, 1), sine.repeat(batch, 1)
    direct = (weighted * cosine[:, None, :]).to(BF16).to(F32)
    turn = (rotated * sine[:, None, :]).to(BF16).to(F32)
    result = (direct + turn).to(BF16)
    flat(query, rows * Q_HEADS * DIM).view(rows, Q_HEADS, DIM).copy_(result[:, :Q_HEADS])
    key_cache = flat(keys, batch * KV_HEADS * CAPACITY * DIM).view(batch, KV_HEADS, CAPACITY, DIM)
    value_cache = flat(values, batch * KV_HEADS * CAPACITY * DIM).view(batch, KV_HEADS, CAPACITY, DIM)
    new_keys = result[:, Q_HEADS:].view(batch, TOKENS, KV_HEADS, DIM)
    new_values = data[:, Q_HEADS + KV_HEADS:].view(batch, TOKENS, KV_HEADS, DIM)
    where = flat(position)
    for b in range(batch):
        for t in range(TOKENS):
            pos = t if PREFILL else int(where[b if ROWS else 0]) + t
            assert 0 <= pos < CAPACITY, f"_qk_rope_cache: row {b} token {t} writes KV slot {pos} of {CAPACITY}"
            key_cache[b, :, pos] = new_keys[b, t]
            value_cache[b, :, pos] = new_values[b, t]


def _interval(query, key, value, visible, scale):
    """One interval's (accumulator, maximum, denominator) for queries [Q,D] over keys [N,D]."""
    count = query.shape[0]
    if key.shape[0] == 0:
        return query.new_zeros((count, query.shape[1]), dtype=F32), torch.full((count,), -math.inf), torch.zeros(count)
    scores = (query.to(F32) @ key.to(F32).T) * (scale * LOG2E)
    scores = torch.where(visible, scores, torch.tensor(-math.inf))
    maximum = scores.max(dim=1).values
    pivot = torch.where(maximum == -math.inf, torch.zeros(()), maximum)
    probabilities = torch.exp2(scores - pivot[:, None])
    return probabilities.to(BF16).to(F32) @ value.to(F32), maximum, probabilities.sum(1)


@reference("_decode_partials")
def _decode_partials(
    grid, q_ptr, k_ptr, v_ptr, position_ptr, partial_ptr, stats_ptr,
    GROUPS, DIM, CAPACITY, SPLITS, CHUNK, SCALE, BLOCK_M, BLOCK_N, **launch,
):
    groups_total = grid[0]
    assert grid[1] == SPLITS and BLOCK_M >= GROUPS and SPLITS * CHUNK >= CAPACITY
    query = flat(q_ptr, groups_total * GROUPS * DIM).view(groups_total, GROUPS, DIM)
    key = flat(k_ptr, groups_total * CAPACITY * DIM).view(groups_total, CAPACITY, DIM)
    value = flat(v_ptr, groups_total * CAPACITY * DIM).view(groups_total, CAPACITY, DIM)
    partial = flat(partial_ptr, groups_total * SPLITS * GROUPS * DIM).view(groups_total, SPLITS, GROUPS, DIM)
    stats = flat(stats_ptr, groups_total * SPLITS * GROUPS * 2).view(groups_total, SPLITS, GROUPS, 2)
    valid = int(flat(position_ptr)[0]) + 1
    assert 1 <= valid <= CAPACITY, f"_decode_partials: valid length {valid} of capacity {CAPACITY}"
    for split in range(SPLITS):
        begin = split * CHUNK
        end = max(begin, min(begin + CHUNK, CAPACITY, valid))
        for g in range(groups_total):
            visible = torch.ones((GROUPS, end - begin), dtype=torch.bool)
            acc, maximum, denominator = _interval(query[g], key[g, begin:end], value[g, begin:end], visible, SCALE)
            partial[g, split], stats[g, split, :, 0], stats[g, split, :, 1] = acc, maximum, denominator


def _merge(partial, stats):
    """[..., S, Q, D] partials and [..., S, Q, 2] stats -> [..., Q, D]."""
    maxima, denominators = stats[..., 0], stats[..., 1]
    maximum = maxima.max(dim=-2, keepdim=True).values
    correction = torch.exp2(maxima - maximum)
    denominator = (denominators * correction).sum(-2)
    numerator = (partial * correction[..., None]).sum(-3)
    return numerator / denominator[..., None]


@reference("_decode_merge")
def _decode_merge(grid, partial_ptr, stats_ptr, out_ptr, GROUPS, DIM, SPLITS, BLOCK_S, **launch):
    assert grid[0] % GROUPS == 0 and BLOCK_S >= SPLITS
    groups_total = grid[0] // GROUPS
    partial = flat(partial_ptr, groups_total * SPLITS * GROUPS * DIM).view(groups_total, SPLITS, GROUPS, DIM)
    stats = flat(stats_ptr, groups_total * SPLITS * GROUPS * 2).view(groups_total, SPLITS, GROUPS, 2)
    flat(out_ptr, grid[0] * DIM).view(groups_total, GROUPS, DIM).copy_(_merge(partial, stats))


@reference("_block_partials")
def _block_partials(
    grid, q_ptr, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
    TOKENS, GROUPS, Q_HEADS, KV_HEADS, DIM, CAPACITY, SPLITS, CHUNK, SCALE, BLOCK_M, BLOCK_N, **launch,
):
    groups_total = grid[0]
    members = TOKENS * GROUPS
    assert grid[1] == SPLITS and BLOCK_M >= members and groups_total % KV_HEADS == 0 and SPLITS * CHUNK >= CAPACITY
    batch = groups_total // KV_HEADS
    query = flat(q_ptr, batch * TOKENS * Q_HEADS * DIM).view(batch, TOKENS, KV_HEADS, GROUPS, DIM)
    key = flat(k_ptr, groups_total * CAPACITY * DIM).view(batch, KV_HEADS, CAPACITY, DIM)
    value = flat(v_ptr, groups_total * CAPACITY * DIM).view(batch, KV_HEADS, CAPACITY, DIM)
    out = flat(out_ptr, batch * TOKENS * Q_HEADS * DIM).view(batch, TOKENS, KV_HEADS, GROUPS, DIM)
    if SPLITS > 1:
        partial = flat(partial_ptr, groups_total * SPLITS * members * DIM).view(batch, KV_HEADS, SPLITS, members, DIM)
        stats = flat(stats_ptr, groups_total * SPLITS * members * 2).view(batch, KV_HEADS, SPLITS, members, 2)
    token = torch.arange(members) // GROUPS
    for row in range(batch):
        first = int(flat(position_ptr)[row]) + 1
        assert first + TOKENS - 1 <= CAPACITY, f"_block_partials: row {row} block ends at slot {first + TOKENS - 2} of {CAPACITY}"
        chained = token < int(flat(chain_ptr)[row])
        valid = first + torch.where(chained, token, torch.zeros_like(token))
        own = torch.where(chained, torch.full_like(token, -1), first - 1 + token)
        for split in range(SPLITS):
            begin = split * CHUNK
            end = max(begin, min(begin + CHUNK, CAPACITY, first + TOKENS - 1))
            slots = torch.arange(begin, end)
            visible = (slots[None, :] < valid[:, None]) | (slots[None, :] == own[:, None])
            for head in range(KV_HEADS):
                q = query[row, :, head].reshape(members, DIM)
                acc, maximum, denominator = _interval(q, key[row, head, begin:end], value[row, head, begin:end], visible, SCALE)
                if SPLITS == 1:
                    out[row, :, head] = (acc / denominator[:, None]).view(TOKENS, GROUPS, DIM).to(out.dtype)
                else:
                    partial[row, head, split] = acc
                    stats[row, head, split, :, 0], stats[row, head, split, :, 1] = maximum, denominator


@reference("_block_partials_tma")
def _block_partials_tma(
    grid, q_ptr, k_ptr, v_ptr, kd_ptr, vd_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
    LIMIT, TMA, CAPACITY, BLOCK_N, **constants,
):
    # Only the ordinary-load twin exists on a CPU; its sums are _block_partials's (reduction order aside).
    assert not TMA and kd_ptr is k_ptr and vd_ptr is v_ptr, "cpu shim: no tensor maps; TMA attention must fail closed"
    assert 0 <= CAPACITY - LIMIT < BLOCK_N
    _block_partials(
        grid, q_ptr, k_ptr, v_ptr, position_ptr, chain_ptr, partial_ptr, stats_ptr, out_ptr,
        CAPACITY=CAPACITY, BLOCK_N=BLOCK_N, **constants,
    )


@reference("_block_merge")
def _block_merge(grid, partial_ptr, stats_ptr, out_ptr, TOKENS, GROUPS, Q_HEADS, KV_HEADS, DIM, SPLITS, BLOCK_S, **launch):
    members = TOKENS * GROUPS
    assert grid[0] % (TOKENS * Q_HEADS) == 0 and BLOCK_S >= SPLITS and Q_HEADS == KV_HEADS * GROUPS
    batch = grid[0] // (TOKENS * Q_HEADS)
    partial = flat(partial_ptr, batch * KV_HEADS * SPLITS * members * DIM).view(batch, KV_HEADS, SPLITS, members, DIM)
    stats = flat(stats_ptr, batch * KV_HEADS * SPLITS * members * 2).view(batch, KV_HEADS, SPLITS, members, 2)
    result = _merge(partial, stats).view(batch, KV_HEADS, TOKENS, GROUPS, DIM).permute(0, 2, 1, 3, 4)
    flat(out_ptr, grid[0] * DIM).view(batch, TOKENS, KV_HEADS, GROUPS, DIM).copy_(result)


def _gemm(grid, x_ptr, weight_ptr, out_ptr, N, K, SPLITS, CHUNK, BLOCK_N, BLOCK_K, M=1, BLOCK_M=None, **launch):
    tiles = launch.get("TILES", 1)  # the hoisted kind covers TILES tiles per program
    assert grid[0] * tiles * BLOCK_N >= N and grid[1] == SPLITS and SPLITS * CHUNK >= K and CHUNK % BLOCK_K == 0
    assert BLOCK_M is None or BLOCK_M >= M
    for flag, truth in (("EVEN_M", M == BLOCK_M), ("EVEN_N", N % (tiles * BLOCK_N) == 0), ("EVEN_K", SPLITS * CHUNK == K)):
        assert not launch.get(flag, False) or truth, f"mask-free launch with a ragged axis: {flag}"
    x = flat(x_ptr, M * K).view(M, K).to(F32)
    weight = flat(weight_ptr, N * K).view(N, K)
    out = flat(out_ptr, SPLITS * M * N).view(SPLITS, M, N)
    for split in range(SPLITS):
        begin, end = split * CHUNK, min((split + 1) * CHUNK, K)
        out[split] = x[:, begin:end] @ weight[:, begin:end].to(F32).T if end > begin else 0


for _name in ("_gemv", "_skinny_gemm", "_exact_gemm", "_trans_gemm", "_hoist_gemm", "_hoist_trans_gemm"):
    reference(_name)(_gemm)


@reference("_persist_trans_gemm")
def _persist_gemm(grid, x_ptr, weight_ptr, out_ptr, M, N, K, PROGRAMS, N_TILES, STEPS, BLOCK_N, BLOCK_K, BLOCK_M, **launch):
    """The persistent 1-D launch: program p owns tiles p, p + PROGRAMS, ...; one split, BF16 [M, N] output."""
    grid = tuple(grid) + (1,) * (3 - len(tuple(grid)))
    assert grid == (PROGRAMS, 1, 1) and 1 <= PROGRAMS <= min(132, N_TILES), f"not a persistent 1-D grid: {grid}"
    assert (N_TILES - 1) * BLOCK_N < N <= N_TILES * BLOCK_N and (STEPS - 1) * BLOCK_K < K <= STEPS * BLOCK_K and BLOCK_M >= M
    for flag, truth in (("EVEN_M", M == BLOCK_M), ("EVEN_N", N % BLOCK_N == 0), ("EVEN_K", K % BLOCK_K == 0)):
        assert not launch.get(flag, False) or truth, f"mask-free launch with a ragged axis: {flag}"
    owners = torch.zeros(N_TILES, dtype=torch.int64)
    for program in range(PROGRAMS):
        owners[program::PROGRAMS] += 1
        assert len(range(program, N_TILES, PROGRAMS)) <= -(-N_TILES // PROGRAMS)
    assert bool((owners == 1).all()), "a tile without exactly one owning program"
    assert out_ptr.dtype == BF16, "one split rounds in the store: the output is BF16, never FP32 partials"
    x = flat(x_ptr, M * K).view(M, K).to(F32)
    weight = flat(weight_ptr, N * K).view(N, K).to(F32)
    flat(out_ptr, M * N).view(M, N).copy_((x @ weight.T).to(BF16))


@reference("_merge_projection")
def _merge_projection(grid, partial_ptr, out_ptr, COUNT, SPLITS, BLOCK_S, BLOCK, **launch):
    assert grid[0] * BLOCK >= COUNT and BLOCK_S >= SPLITS
    flat(out_ptr, COUNT).copy_(flat(partial_ptr, SPLITS * COUNT).view(SPLITS, COUNT).sum(0))


@reference("_paired_projection")
def _paired_projection(grid, x_ptr, weight_ptr, out_ptr, M, I, K, BM, BN, BK, **launch):
    assert grid[0] == -(-M // BM) * -(-I // BN)
    x = flat(x_ptr, M * K).view(M, K).to(F32)
    weight = flat(weight_ptr, 2 * I * K).view(2 * I, K).to(F32)
    gate = (x @ weight[:I].T).to(BF16).to(F32)
    up = (x @ weight[I:].T).to(BF16).to(F32)
    activated = (gate / (1.0 + torch.exp(-gate))).to(BF16).to(F32)
    flat(out_ptr, M * I).view(M, I).copy_(activated * up)


def _emulation(name, cache={}):
    """One function of ../check_tree.py (a script: importing it would run its own cases)."""
    if name not in cache:
        import os
        source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "check_tree.py")).read()
        scope = {}
        exec(source[:source.index("def make_model")], scope)
        cache.update(scope)
    return cache[name]


@reference("_propose")
def _propose(
    grid, history, position, successor, stale, tokens, chains, phases,
    SIZE, TOKENS, MAXLEN, D0, D1, D2, D3, LANES, ALTERNATES, TOP, BLOCK, BLOCK_S, BLOCK_T, **launch,
):
    """The shipped drafting kernel, through check_tree.py's line-by-line emulation."""
    emu_propose = _emulation("emu_propose")
    assert BLOCK >= SIZE and BLOCK_T >= TOKENS
    assert LANES == TOKENS - 1 - min(D0, D1, D2, D3) and ALTERNATES == min(LANES, 3)
    table = flat(successor)

    class Top:
        def __getitem__(self, token):
            return table[token * TOP:(token + 1) * TOP].tolist()

    for row in range(grid[0]):
        h = flat(history, (row + 1) * SIZE)[row * SIZE:].tolist()
        place = int(flat(position)[row])
        assert 0 <= place < SIZE, f"_propose: row {row} position {place} outside history of {SIZE}"
        out, chain, phase = emu_propose(h, place, Top(), SIZE, TOKENS, (D0, D1, D2, D3), MAXLEN, TOP, int(flat(stale)[row]))
        flat(tokens, (row + 1) * TOKENS)[row * TOKENS:].copy_(torch.tensor(out, dtype=torch.int64))
        flat(chains)[row] = chain
        flat(phases, (row + 1) * TOKENS)[row * TOKENS:].copy_(torch.tensor(phase, dtype=torch.int64))


@reference("_propose_ranked")
def _propose_ranked(
    grid, history, position, successor, weights, tokens, chains, phases,
    SIZE, TOKENS, MAXLEN, D0, D1, D2, D3, HIST, TOP, BLOCK, BLOCK_C, BLOCK_T, **launch,
):
    assert BLOCK >= SIZE and BLOCK_T >= TOKENS
    table = flat(successor)
    w = flat(weights, 21).tolist()
    NEG = -math.inf
    for row in range(grid[0]):
        h = flat(history, (row + 1) * SIZE)[row * SIZE:].tolist()
        place = int(flat(position)[row])
        assert 0 <= place < SIZE, f"_propose_ranked: row {row} position {place} outside history of {SIZE}"
        last = h[place]
        one = [i < place and h[i] == last for i in range(SIZE)]
        agree, length = list(one), [int(o) for o in one]
        for back in range(1, MAXLEN):
            wanted = h[max(place - back, 0)]
            for i in range(SIZE):
                agree[i] = agree[i] and i >= back and place >= back and h[i - back] == wanted
                length[i] += int(agree[i])
        rank = [i + (length[i] - 1) * SIZE if one[i] else -1 for i in range(SIZE)]
        after = [h[i + 1] if one[i] else -1 for i in range(SIZE)]
        candidate, count, open_rank = [-1] * BLOCK_C, 0, list(rank)
        for _ in range(HIST):
            choice = max(open_rank)
            usable = choice >= 0
            token = h[max(choice, 0) % SIZE + 1]
            if usable and count < BLOCK_C:
                candidate[count] = token
            open_rank = [-1 if (one[i] and after[i] == token) else open_rank[i] for i in range(SIZE)]
            count += int(usable)
        table_rank = [TOP] * BLOCK_C
        for entry in range(TOP):
            token = int(table[last * TOP + entry])
            known = [candidate[l] == token and l < count for l in range(BLOCK_C)]
            table_rank = [entry if known[l] else table_rank[l] for l in range(BLOCK_C)]
            add = not any(known) and count < BLOCK_C
            if add:
                candidate[count], table_rank[count] = token, entry
            count += int(add)
        score, suffix, origin = [], [], []
        for lane in range(BLOCK_C):
            follows = [one[i] and after[i] == candidate[lane] for i in range(SIZE)]
            best_rank = max([rank[i] if follows[i] else -1 for i in range(SIZE)])
            sfx = best_rank // SIZE + 1 if best_rank >= 0 else 0
            occurs = sum(1 for i in range(SIZE) if i <= place and h[i] == candidate[lane])
            value = NEG
            if lane < count:
                value = w[sfx] + w[9 + table_rank[lane]] + (w[18] if lane == 0 and sfx > 0 else 0.0)
                value += w[19] * math.log(1.0 + sum(follows)) + w[20] * math.log(1.0 + occurs)
            score.append(value)
            suffix.append(sfx)
            origin.append(best_rank % SIZE if best_rank >= 0 else 0)
        top = max(range(BLOCK_C), key=lambda lane: (score[lane], -lane))
        first, matched, start = candidate[top], suffix[top], origin[top]
        score[top] = NEG
        drafts = D0 if matched <= 1 else D1 if matched <= 3 else D2 if matched <= 7 else D3
        out = flat(tokens, (row + 1) * TOKENS)[row * TOKENS:]
        out[0], out[1] = last, first
        previous = first
        for step in range(2, TOKENS):
            source = start + step
            copied = h[min(source, SIZE - 1)]
            draft = copied if (matched > 0 and source <= place) else int(table[previous * TOP])
            if drafts >= step:
                out[step] = draft
            previous = draft
        for lane in range(TOKENS - 2):
            pick = max(range(BLOCK_C), key=lambda c: (score[c], -c))
            token = candidate[pick] if score[pick] > NEG else first
            if 1 + drafts + lane < TOKENS:
                out[1 + drafts + lane] = token
            score[pick] = NEG
        flat(chains)[row] = 1 + drafts
        phase_row = flat(phases, (row + 1) * TOKENS)[row * TOKENS:]
        for slot in range(TOKENS):
            phase_row[slot] = slot if slot <= drafts else 1


@reference("_settle")
def _settle(grid, tokens, greedy, position, limit, history, result, move_from, move_to, chains, stale, SIZE, TOKENS, BLOCK, **launch):
    assert BLOCK >= TOKENS
    for row in range(grid[0]):
        CHAIN = int(flat(chains)[row])
        draft = flat(tokens, (row + 1) * TOKENS)[row * TOKENS:].tolist()
        chosen = flat(greedy, (row + 1) * TOKENS)[row * TOKENS:].tolist()
        miss = [slot if (1 <= slot < CHAIN and draft[slot] != chosen[slot - 1]) else CHAIN for slot in range(TOKENS)]
        gained = min(miss)
        wanted = chosen[0]
        hit = min([slot if (slot >= CHAIN and draft[slot] == wanted) else TOKENS for slot in range(TOKENS)])
        place = int(flat(position)[row])
        room = max(int(flat(limit)[row]) - place, 0)
        branch = gained == 1 and hit < TOKENS and room >= 2
        flat(stale)[row] = chosen[min(gained, TOKENS - 1)] if (gained < CHAIN and not branch) else -1
        gained = min(2 if branch else gained, room)
        bonus = chosen[min(hit, TOKENS - 1)]
        emitted = [bonus if (branch and slot == 1) else chosen[slot] for slot in range(TOKENS)]
        out = flat(result, (row + 1) * (TOKENS + 1))[row * (TOKENS + 1):]
        out[0] = gained
        out[1:] = torch.tensor(emitted)
        assert place + 1 + TOKENS <= SIZE, f"_settle: row {row} history write ends at {place + TOKENS} of {SIZE}"
        flat(history, (row + 1) * SIZE)[row * SIZE + place + 1: row * SIZE + place + 1 + TOKENS] = torch.tensor(emitted)
        flat(move_from)[row] = place + hit if branch else -1
        flat(move_to)[row] = place + 1
        flat(position)[row] = place + gained


RELOCATIONS = [0]


@reference("_relocate")
def _relocate(grid, store, move_from, move_to, BATCH, KV_HEADS, CAPACITY, DIM, BLOCK_H, **launch):
    assert grid[1] == BATCH and BLOCK_H >= KV_HEADS
    planes = flat(store, grid[0] * BATCH * KV_HEADS * CAPACITY * DIM).view(grid[0], BATCH, KV_HEADS, CAPACITY, DIM)
    for row in range(BATCH):
        source, target = int(flat(move_from)[row]), int(flat(move_to)[row])
        if source >= 0:
            assert source < CAPACITY and 0 <= target < CAPACITY, f"_relocate: {source}->{target} of {CAPACITY}"
            planes[:, row, :, target] = planes[:, row, :, source].clone()
            RELOCATIONS[0] += 1
