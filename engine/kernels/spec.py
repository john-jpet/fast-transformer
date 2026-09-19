"""Fused bookkeeping for exact speculation: one launch to propose, one to settle.

These are the tensor programs of ``speculate.propose`` and
``speculate.accept``/``advance`` (which remain the readable reference and are
what the CPU tests exercise), written as one Triton program per row so that a
verify pass does not pay for some sixty tiny elementwise launches.
"""

import torch
import triton
import triton.language as tl

from kernels.pdl import wait as pdl_wait


@triton.jit
def _propose(
    history, position, successor, stale, tokens, chains, phases,
    SIZE: tl.constexpr, TOKENS: tl.constexpr, MAXLEN: tl.constexpr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
    LANES: tl.constexpr, ALTERNATES: tl.constexpr,
    TOP: tl.constexpr, BLOCK: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Row layout: trusted token, D chain drafts, TOKENS - 1 - D alternatives to draft 1.

    D depends on how long a suffix of the row was found earlier in it (0-1,
    2-3, 4-7 or 8+ tokens -> D0..D3): a long match earns a deep chain, no match
    earns wide alternatives. ``chains`` gets 1 + D, ``phases`` each token's
    RoPE offset (the chain counts up; alternatives stand where draft 1 stands).
    """
    pdl_wait()  # before any global memory access
    row = tl.program_id(0).to(tl.int64)
    base = history + row * SIZE
    place = tl.load(position + row)
    last = tl.load(base + place)
    index = tl.arange(0, BLOCK).to(tl.int64)
    inside = index < SIZE
    here = tl.load(base + index, inside, other=-1)
    # Earlier occurrences of the newest token, and how far back each agrees.
    one = inside & (index < place) & (here == last)
    agree = one
    length = one.to(tl.int64)
    for back in tl.static_range(1, MAXLEN):
        wanted = tl.load(base + tl.maximum(place - back, 0))
        seen = tl.load(base + index - back, inside & (index >= back), other=-1)
        agree = agree & (index >= back) & (place >= back) & (seen == wanted)
        length += agree.to(tl.int64)
    # Longer suffix first, then the most recent occurrence.
    rank = tl.where(one, index + (length - 1) * SIZE, -1)
    best = tl.max(rank, axis=0)
    found = best >= 0
    start = tl.where(found, best % SIZE, place)
    matched = tl.where(found, best // SIZE + 1, 0)
    drafts = tl.where(matched <= 1, D0, tl.where(matched <= 3, D1, tl.where(matched <= 7, D2, D3)))
    out = tokens + row * TOKENS
    tl.store(out, last)
    previous = last
    first = last
    for step in tl.static_range(1, TOKENS):
        source = start + step
        copied = tl.load(base + tl.minimum(source, SIZE - 1))
        followed = tl.load(successor + previous * TOP)
        draft = tl.where(found & (source <= place), copied, followed)
        tl.store(out + step, draft, mask=drafts >= step)
        previous = draft
        if step == 1:
            first = draft
    if LANES > 0:
        # Other candidates for draft 1: what followed other occurrences of the
        # newest token, then the table's next choices; all distinct from draft 1.
        lanes = tl.arange(0, BLOCK_S)
        siblings = tl.full((BLOCK_S,), -1, tl.int64)
        count = tl.zeros((), tl.int32)
        after = tl.load(base + index + 1, one, other=-1)
        # First alternative: what the model itself predicted at this position
        # in the previous pass, right after that pass's first wrong draft (one
        # token of its context was wrong, so it is a guess, never trusted).
        # Offline on the model's greedy text: 2-3% fewer passes at 16 tokens.
        hint = tl.load(stale + row)
        seeded = (hint >= 0) & (hint != first) & (TOKENS - 1 - drafts >= 2)
        siblings = tl.where((lanes == 0) & seeded, hint, siblings)
        count += seeded.to(tl.int32)
        for _ in tl.static_range(ALTERNATES):
            taken = (after == first) | (tl.sum((after[:, None] == siblings[None, :]).to(tl.int32), axis=1) > 0)
            choice = tl.max(tl.where(one & (taken == 0), rank, -1), axis=0)
            usable = (choice >= 0) & (count < LANES)
            candidate = tl.load(base + tl.maximum(choice, 0) % SIZE + 1)
            siblings = tl.where((lanes == count) & usable, candidate, siblings)
            count += usable.to(tl.int32)
        for entry in tl.static_range(TOP):
            candidate = tl.load(successor + last * TOP + entry)
            fresh = (candidate != first) & (tl.sum((siblings == candidate).to(tl.int32), axis=0) == 0)
            usable = fresh & (count < LANES)
            siblings = tl.where((lanes == count) & usable, candidate, siblings)
            count += usable.to(tl.int32)
        # An unfilled lane repeats draft 1, which can never be accepted twice.
        tl.store(
            out + 1 + drafts + lanes, tl.where(siblings >= 0, siblings, first),
            (lanes < LANES) & (1 + drafts + lanes < TOKENS),
        )
    slot = tl.arange(0, BLOCK_T).to(tl.int64)
    tl.store(chains + row, 1 + drafts)
    tl.store(phases + row * TOKENS + slot, tl.where(slot <= drafts, slot, 1), slot < TOKENS)


def propose(history, position, tokens_per_row, drafts_by_match, successor, stale, chains, phases):
    """Per row: trusted token, chain drafts, alternatives (int64 [B, T]); fills ``chains`` and ``phases``."""
    batch, size = history.shape
    assert history.is_contiguous() and history.dtype == position.dtype == successor.dtype == torch.int64
    assert position.shape == (batch,) and successor.dim() == 2 and successor.is_contiguous()
    assert chains.shape == (batch,) and phases.shape == (batch, tokens_per_row) and phases.is_contiguous()
    assert stale.shape == (batch,) and stale.dtype == torch.int64
    assert len(drafts_by_match) == 4 and all(1 <= d <= tokens_per_row - 1 for d in drafts_by_match)
    lanes = tokens_per_row - 1 - min(drafts_by_match)
    tokens = torch.empty((batch, tokens_per_row), dtype=torch.int64, device=history.device)
    _propose[(batch,)](
        history, position, successor, stale, tokens, chains, phases,
        SIZE=size, TOKENS=tokens_per_row, MAXLEN=8,
        D0=drafts_by_match[0], D1=drafts_by_match[1], D2=drafts_by_match[2], D3=drafts_by_match[3],
        LANES=lanes, ALTERNATES=min(lanes, 3), TOP=successor.shape[1],
        BLOCK=triton.next_power_of_2(size), BLOCK_S=triton.next_power_of_2(max(lanes, 1)),
        BLOCK_T=triton.next_power_of_2(tokens_per_row), num_warps=4,
    )
    return tokens


@triton.jit
def _settle(
    tokens, greedy, position, limit, history, result, move_from, move_to, chains, stale,
    SIZE: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr,
):
    pdl_wait()  # before any global memory access
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BLOCK).to(tl.int64)
    inside = slot < TOKENS
    CHAIN = tl.load(chains + row)
    draft = tl.load(tokens + row * TOKENS + slot, inside, other=0)
    chosen = tl.load(greedy + row * TOKENS + slot, inside, other=0)
    expected = tl.load(greedy + row * TOKENS + slot - 1, inside & (slot >= 1), other=0)
    # Chain draft ``slot`` stands only if it equals the model's choice after
    # the tokens before it; the first miss ends the run. Gained = kept + 1.
    miss = tl.where(inside & (slot >= 1) & (slot < CHAIN) & (draft != expected), slot, CHAIN)
    gained = tl.min(miss, axis=0)
    # If draft 1 missed, an alternative equal to the model's first choice is
    # kept instead; the model's choice after it is the second gained token.
    wanted = tl.load(greedy + row * TOKENS)
    hit = tl.min(tl.where(inside & (slot >= CHAIN) & (draft == wanted), slot, TOKENS), axis=0)
    place = tl.load(position + row)
    room = tl.maximum(tl.load(limit + row) - place, 0)
    branch = (gained == 1) & (hit < TOKENS) & (room >= 2)
    # After a wrong chain draft the model's output at that slot predicts the
    # token after the new trusted one: the next pass tries it as an alternative.
    guess = tl.load(greedy + row * TOKENS + tl.minimum(gained, TOKENS - 1))
    tl.store(stale + row, tl.where((gained < CHAIN) & (branch == 0), guess, -1))
    gained = tl.minimum(tl.where(branch, 2, gained), room)
    bonus = tl.load(greedy + row * TOKENS + tl.minimum(hit, TOKENS - 1))
    emitted = tl.where(branch & (slot == 1), bonus, chosen)
    tl.store(result + row * (TOKENS + 1), gained)
    tl.store(result + row * (TOKENS + 1) + 1 + slot, emitted, inside)
    # Entries past ``gained`` are rewritten by the next pass before any read.
    tl.store(history + row * SIZE + place + 1 + slot, emitted, inside)
    # The kept alternative's K/V sits in its own slot; it belongs at place + 1.
    tl.store(move_from + row, tl.where(branch, place + hit, -1))
    tl.store(move_to + row, place + 1)
    tl.store(position + row, place + gained)


def settle(tokens, greedy, position, limit, history, result, move_from, move_to, chains, stale):
    """Accept, clamp at ``limit``, record emitted tokens, plan the KV move, advance; in place."""
    batch, count = tokens.shape
    assert tokens.is_contiguous() and greedy.is_contiguous() and history.is_contiguous() and result.is_contiguous()
    assert greedy.shape == tokens.shape and result.shape == (batch, count + 1) and chains.shape == (batch,)
    assert tokens.dtype == greedy.dtype == position.dtype == limit.dtype == history.dtype == result.dtype == torch.int64
    assert move_from.dtype == move_to.dtype == torch.int64 and move_from.shape == move_to.shape == (batch,)
    assert stale.shape == (batch,) and stale.dtype == torch.int64
    _settle[(batch,)](
        tokens, greedy, position, limit, history, result, move_from, move_to, chains, stale,
        SIZE=history.shape[1], TOKENS=count, BLOCK=triton.next_power_of_2(count), num_warps=1,
    )


@triton.jit
def _relocate(
    store, move_from, move_to,
    BATCH: tl.constexpr, KV_HEADS: tl.constexpr, CAPACITY: tl.constexpr, DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pdl_wait()  # before any global memory access
    plane = tl.program_id(0).to(tl.int64)  # flattened (key/value, layer)
    row = tl.program_id(1).to(tl.int64)
    source = tl.load(move_from + row)
    target = tl.load(move_to + row)
    heads = tl.arange(0, BLOCK_H).to(tl.int64)[:, None]
    dims = tl.arange(0, DIM).to(tl.int64)[None, :]
    live = (heads < KV_HEADS) & (source >= 0)
    base = store + ((plane * BATCH + row) * KV_HEADS + heads) * CAPACITY * DIM
    moved = tl.load(base + tl.maximum(source, 0) * DIM + dims, live, other=0)
    tl.store(base + target * DIM + dims, moved, live)


def relocate(store, move_from, move_to):
    """Copy one K/V slot per row, in every layer: store is [2, L, B, Hkv, C, D]."""
    planes, layers, batch, kv_heads, capacity, dim = store.shape
    assert store.is_contiguous() and planes == 2 and move_from.shape == move_to.shape == (batch,)
    _relocate[(planes * layers, batch)](
        store, move_from, move_to,
        BATCH=batch, KV_HEADS=kv_heads, CAPACITY=capacity, DIM=dim,
        BLOCK_H=triton.next_power_of_2(kv_heads), num_warps=1,
    )
