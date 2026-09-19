"""Emulate kernels/spec.py (tree propose + settle) line by line; the emitted text must equal sequential greedy."""
import random

def emu_propose(h, place, top, SIZE, TOKENS, D, MAXLEN, TOP, hint=-1):
    LANES = TOKENS - 1 - min(D); ALTERNATES = min(LANES, 3)
    last = h[place]; rank = []; one_mask = []
    for index in range(SIZE):
        one = index < place and h[index] == last
        agree = one; length = 1 if one else 0
        for back in range(1, MAXLEN):
            wanted = h[max(place - back, 0)]; seen = h[index - back] if index >= back else -1
            agree = agree and index >= back and place >= back and seen == wanted
            length += 1 if agree else 0
        rank.append(index + (length - 1) * SIZE if one else -1); one_mask.append(one)
    best = max(rank); found = best >= 0; start = best % SIZE if found else place
    matched = best // SIZE + 1 if found else 0
    drafts = D[0] if matched <= 1 else D[1] if matched <= 3 else D[2] if matched <= 7 else D[3]
    out = [None] * TOKENS; out[0] = last; previous = last; first = last
    for step in range(1, TOKENS):
        source = start + step
        copied = h[min(source, SIZE - 1)]; followed = top[previous][0]
        draft = copied if (found and source <= place) else followed
        if drafts >= step: out[step] = draft
        previous = draft
        if step == 1: first = draft
    if LANES > 0:
        siblings = [-1] * LANES; count = 0
        if hint >= 0 and hint != first and TOKENS - 1 - drafts >= 2: siblings[0] = hint; count = 1
        after = [h[i + 1] if one_mask[i] else -1 for i in range(SIZE)]
        for _ in range(ALTERNATES):
            choice = max((rank[i] if (one_mask[i] and not (after[i] == first or after[i] in siblings)) else -1) for i in range(SIZE))
            usable = choice >= 0 and count < LANES
            candidate = h[max(choice, 0) % SIZE + 1]
            if usable: siblings[count] = candidate; count += 1
        for entry in range(TOP):
            candidate = top[last][entry]
            usable = candidate != first and candidate not in siblings and count < LANES
            if usable: siblings[count] = candidate; count += 1
        for lane in range(LANES):
            if 1 + drafts + lane < TOKENS: out[1 + drafts + lane] = siblings[lane] if siblings[lane] >= 0 else first
    assert None not in out
    phases = [slot if slot <= drafts else 1 for slot in range(TOKENS)]
    return out, 1 + drafts, phases

def emu_settle(tokens, greedy, place, limit, CHAIN):
    T = len(tokens)
    miss = [slot if (1 <= slot < CHAIN and tokens[slot] != greedy[slot - 1]) else CHAIN for slot in range(T)]
    gained = min(miss); wanted = greedy[0]
    hit = min([slot if (slot >= CHAIN and tokens[slot] == wanted) else T for slot in range(T)])
    room = max(limit - place, 0)
    branch = gained == 1 and hit < T and room >= 2
    emu_settle.stale = greedy[min(gained, T - 1)] if (gained < CHAIN and not branch) else -1
    gained = min(2 if branch else gained, room)
    bonus = greedy[min(hit, T - 1)]
    emitted = [bonus if (branch and slot == 1) else greedy[slot] for slot in range(T)]
    return gained, emitted, (place + hit if branch else -1), place + 1

def make_model(seed, vocab, repeat):
    rng = random.Random(seed); table = {}
    def nxt(prefix):
        key = tuple(prefix[-2:]) if repeat else (len(prefix), prefix[-1])
        if key not in table: table[key] = rng.randrange(vocab)
        return table[key]
    return nxt

stats = {"branch": 0, "passes": 0, "chains": set()}
for seed in range(400):
    rng = random.Random(seed); repeat = seed % 2 == 0; vocab = 5 if repeat else 12
    TOKENS, D = rng.choice([(16, (5, 8, 13, 14)), (8, (2, 4, 6, 7)), (5, (2, 3, 4, 4)), (4, (1, 2, 3, 3)), (3, (1, 2, 2, 2)), (2, (1, 1, 1, 1))]); TOP = 8
    prompt_len, outputs = rng.choice([(1, 6), (3, 12), (9, 32), (30, 40)])
    nxt = make_model(seed, vocab, repeat)
    top = [[rng.randrange(vocab) for _ in range(TOP)] for _ in range(vocab)]
    prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
    ref = list(prompt)
    for _ in range(outputs): ref.append(nxt(ref))
    SIZE = prompt_len + outputs + TOKENS + 2; limit = prompt_len + outputs - 1
    h = [0] * SIZE; h[:prompt_len] = prompt; h[prompt_len] = nxt(prompt); place = prompt_len
    emitted_all = [h[prompt_len]]
    kv = {i: prompt[i] for i in range(prompt_len)}     # slot -> token whose K/V it holds
    hint = -1
    while len(emitted_all) < outputs:
        tokens, CHAIN, phases = emu_propose(h, place, top, SIZE, TOKENS, D, 8, TOP, hint)
        if hint >= 0 and hint in tokens[CHAIN:]: stats['hints'] = stats.get('hints', 0) + 1
        stats["chains"].add((TOKENS, CHAIN))
        assert tokens[0] == h[place] and phases[:CHAIN] == list(range(CHAIN)) and all(p == 1 for p in phases[CHAIN:])
        for t in range(TOKENS): kv[place + t] = tokens[t]
        assert [kv[i] for i in range(place)] == h[:place], "cache prefix must hold exactly the known text"
        known = h[:place]; greedy = []
        for t in range(TOKENS):
            context = known + tokens[: t + 1] if t < CHAIN else known + [tokens[0], tokens[t]]   # tree mask
            greedy.append(nxt(context))
        gained, emitted, move_from, move_to = emu_settle(tokens, greedy, place, limit, CHAIN); hint = emu_settle.stale
        if move_from >= 0: kv[move_to] = kv[move_from]; stats["branch"] += 1
        for slot in range(TOKENS): h[place + 1 + slot] = emitted[slot]
        emitted_all += emitted[:gained]; place += gained; stats["passes"] += 1
        assert place <= limit and place + TOKENS < SIZE
    assert emitted_all == ref[prompt_len:], (seed, emitted_all[:10], ref[prompt_len:prompt_len + 10])
print(f"tree speculation with match-length shapes equals sequential greedy in 400 cases ({stats.get('hints', 0)} stale-guess alternatives, {stats['branch']} alternative branches in {stats['passes']} passes; {len(stats['chains'])} distinct (tokens, chain) shapes seen); cache prefix invariant held")
