"""``_block_partials_tma`` with ordinary loads (TMA off) vs ``_block_partials``: bit-identical attention.

The twin turns the prefix product (``tl.dot(key, query^T)``, statistics along axis 0, transposed computed
probabilities into the V product) so that a tensor-map tile can be its first operand. Same tiles, same order,
same arithmetic: the outputs must be torch.equal for every layout, with the plain kernel's prefix loop on and
off. Also: a "tma" layout without tensor maps (this CPU) must launch the plain kernel. Exit status 1 on failure.
"""
import interp_bf16, itertools, sys, torch
from kernels import decode_attention as A

torch.manual_seed(0)
failures = 0
CASES = (
    # B, T, Hq, Hkv, D, C
    (2, 4, 4, 2, 128, 300), (1, 5, 8, 2, 128, 549), (3, 2, 8, 8, 64, 200), (2, 16, 4, 1, 128, 260), (2, 8, 4, 2, 128, 131),
)
LAYOUTS = ((64, 1, 4), (128, 1, 4), (64, 4, 4), (32, 3, 4), (128, 2, 4))
for (B, T, Hq, Hkv, D, C), layout in itertools.product(CASES, LAYOUTS):
    q = torch.randn(B, T, Hq, D).bfloat16()
    k = (torch.randn(B, Hkv, C, D) * 2).bfloat16()
    v = torch.randn(B, Hkv, C, D).bfloat16()
    # One row near the end of the cache (LIMIT matters there), one short, the rest random.
    position = torch.randint(0, C - T, (B,))
    position[0] = C - T - 1
    if B > 1:
        position[1] = 3
    for chain in (T, 2, 1):
        chains = torch.full((B,), chain, dtype=torch.int64)
        if B > 1:
            chains[1] = 1
        twin = A._launch_block("twin", q, k, v, position, chains, D ** -0.5, layout)
        for prefix in (True, False):
            A._wide_prefix = lambda batch, capacity, prefix=prefix: prefix
            plain = A._launch_block("plain", q, k, v, position, chains, D ** -0.5, layout)
            same = torch.equal(twin, plain) and bool(torch.isfinite(twin.float()).all())
            failures += not same
            print(f"{'PASS' if same else 'FAIL'} twin == plain(PREFIX={prefix}) B{B} T{T} Hq{Hq} Hkv{Hkv} D{D} C{C} chain={chain} "
                  f"layout={layout} max|diff|={float((twin.float() - plain.float()).abs().max()):.3e}", flush=True)
        A._wide_prefix = lambda batch, capacity: True
        A._BLOCK_LAYOUTS[(q.device, B, T, Hq, Hkv, C, D)] = layout + (A.TMA,)
        closed = A.block_attention(q, k, v, position, D ** -0.5, chains)
        same = torch.equal(closed, A._launch_block("plain", q, k, v, position, chains, D ** -0.5, layout))
        failures += not same
        print(f"{'PASS' if same else 'FAIL'} 'tma' layout without tensor maps == plain launch", flush=True)
# The launcher's plumbing with stand-in tensor maps (the twin plays the TMA kernel): probe, verdict cache, launch.
real_launch, real_maps = A._launch_block, A._tma_maps
calls = []


def launch(kind, query, key, value, position, chain, scale, layout, maps=None):
    calls.append(kind)
    if kind == "tma":
        assert maps[0] is key and maps[1] is value and maps[2] == key.shape[2] - key.shape[0] * key.shape[1] * key.shape[2] % layout[0]
        kind = "twin"
    return real_launch(kind, query, key, value, position, chain, scale, layout[:3])


A._launch_block = launch
A._tma_maps = lambda key, value, block_n: (key, value, key.shape[2] - key.shape[0] * key.shape[1] * key.shape[2] % block_n)
A._wide_prefix = lambda batch, capacity: True
B, T, Hq, Hkv, D, C = 2, 4, 4, 2, 128, 300
q = torch.randn(B, T, Hq, D).bfloat16(); k = torch.randn(B, Hkv, C, D).bfloat16(); v = torch.randn(B, Hkv, C, D).bfloat16()
position, chains = torch.tensor([C - T, 100]), torch.tensor([T, 2])
for layout in ((64, 2, 4, A.TMA, 2), (64, 2, 4, A.TMA, 3)):
    A._BLOCK_LAYOUTS[(q.device, B, T, Hq, Hkv, C, D)] = layout
    for attempt in range(2):
        calls.clear()
        got = A.block_attention(q, k, v, position, D ** -0.5, chains)
        want = real_launch("plain", q, k, v, position, chains, D ** -0.5, layout[:3])
        expected = ["tma", "plain", "tma"] if attempt == 0 else ["tma"]  # probe once, then launches only
        same = torch.equal(got, want) and calls == expected and not A._TMA_ATTENTION_OFF[0]
        failures += not same
        print(f"{'PASS' if same else 'FAIL'} launcher with stand-in maps {layout} attempt {attempt}: calls={calls}", flush=True)
A._tma_maps = lambda key, value, block_n: None
calls.clear()
A._TMA_CHECKED.clear()
got = A.block_attention(q, k, v, position, D ** -0.5, chains)
same = torch.equal(got, want) and calls == ["plain"]
failures += not same
print(f"{'PASS' if same else 'FAIL'} no tensor maps: calls={calls}", flush=True)
A._launch_block, A._tma_maps = real_launch, real_maps
print("failures", failures)
sys.exit(1 if failures else 0)
