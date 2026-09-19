"""Run engine/kernels/spec.py (_propose, _settle, _relocate) in the Triton interpreter vs agent/local_cpu/check_tree.py emulation."""
import interp_bf16, random, sys, io, contextlib, torch
with contextlib.redirect_stdout(io.StringIO()):
    import check_tree as emu            # its self-test runs on import
from kernels import spec

rng = random.Random(1); cases = 0; branches = 0
for case in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    TOKENS, D = rng.choice([(16, (5, 8, 13, 14)), (8, (2, 4, 6, 7)), (5, (2, 3, 4, 4)), (4, (1, 2, 3, 3)), (2, (1, 1, 1, 1))])
    batch, vocab, SIZE, TOP = rng.choice([1, 3, 4]), rng.choice([4, 9, 40]), rng.choice([40, 97]), 8
    top = torch.tensor([[rng.randrange(vocab) for _ in range(TOP)] for _ in range(vocab)])
    history = torch.zeros(batch, SIZE, dtype=torch.int64); position = torch.zeros(batch, dtype=torch.int64)
    for b in range(batch):
        place = rng.randrange(0, SIZE - TOKENS - 2); position[b] = place
        history[b, :place + 1] = torch.tensor([rng.randrange(vocab) for _ in range(place + 1)])
    chains = torch.zeros(batch, dtype=torch.int64); phases = torch.zeros(batch, TOKENS, dtype=torch.int64)
    stale = torch.tensor([rng.choice([-1, -1, rng.randrange(vocab)]) for _ in range(batch)], dtype=torch.int64)
    tokens = spec.propose(history, position, TOKENS, D, top, stale, chains, phases)
    greedy = torch.tensor([[rng.randrange(vocab) for _ in range(TOKENS)] for _ in range(batch)])
    want = []
    for b in range(batch):
        out, chain, ph = emu.emu_propose(history[b].tolist(), int(position[b]), top.tolist(), SIZE, TOKENS, D, 8, TOP, int(stale[b]))
        assert tokens[b].tolist() == out and int(chains[b]) == chain and phases[b].tolist() == ph, ("propose", case, b, tokens[b].tolist(), out)
        for i in range(1, TOKENS):                                  # make drafts agree sometimes
            if rng.random() < 0.6: greedy[b, i - 1] = out[i]
        if rng.random() < 0.5 and chain < TOKENS: greedy[b, 0] = out[rng.randrange(chain, TOKENS)]
    limit = position + torch.tensor([rng.randrange(0, TOKENS + 3) for _ in range(batch)])
    h2, p2 = history.clone(), position.clone(); result = torch.zeros(batch, TOKENS + 1, dtype=torch.int64)
    move_from = torch.zeros(batch, dtype=torch.int64); move_to = torch.zeros(batch, dtype=torch.int64)
    spec.settle(tokens, greedy, p2, limit, h2, result, move_from, move_to, chains, stale)
    for b in range(batch):
        place = int(position[b])
        gained, emitted, mf, mt = emu.emu_settle(tokens[b].tolist(), greedy[b].tolist(), place, int(limit[b]), int(chains[b]))
        assert (int(result[b, 0]), result[b, 1:].tolist(), int(move_from[b]), int(move_to[b]), int(p2[b])) == (gained, emitted, mf, mt, place + gained), ("settle", case, b)
        assert h2[b, place + 1: place + 1 + TOKENS].tolist() == emitted
        assert int(stale[b]) == emu.emu_settle.stale, ("stale", case, b)
        branches += mf >= 0
    # _relocate on a bf16 [2, L, B, Hkv, C, D] store
    L, Hkv, Dm = 2, 3, 4
    store = torch.randn(2, L, batch, Hkv, SIZE, Dm).bfloat16(); ref = store.clone()
    for b in range(batch):
        if int(move_from[b]) >= 0: ref[:, :, b, :, int(move_to[b])] = ref[:, :, b, :, int(move_from[b])]
    spec.relocate(store, move_from, move_to)
    assert torch.equal(store, ref), ("relocate", case)
    cases += 1
print(f"spec.py in Triton interpreter == check_tree emulation: {cases} cases (_propose, _settle, _relocate; {branches} alternative branches)")
