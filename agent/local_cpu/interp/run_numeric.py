"""BF16 kernels (rmsnorm, swiglu, linear, decode/block attention) in the Triton interpreter vs torch / emulation references."""
import interp_bf16, ast, math, sys, traceback, torch
from torch.nn import functional as F
torch.manual_seed(0)
def report(name, got, want, tol=None):
    diff = (got.float() - want.float()).abs(); exact = torch.equal(got, want)
    same = float((got == want).float().mean())
    ok = exact if tol is None else bool((diff <= tol).all())
    print(f"{'PASS' if ok else 'FAIL'} {name}: bit-exact={exact} equal-frac={same:.4f} max|diff|={float(diff.max()):.3e}", flush=True)
def section(fn):
    try: fn()
    except Exception: print(f"ERROR {fn.__name__}:\n" + traceback.format_exc(limit=6), flush=True)

def rmsnorm():
    from kernels.rmsnorm import rms_norm, add_rms_norm
    def ref(x, w, eps):
        v = x.float(); v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return w * v.to(x.dtype)
    for shape in ((3, 2560), (2, 5, 128), (1, 100)):
        x = (torch.randn(shape) * 3).bfloat16(); w = torch.randn(shape[-1]).bfloat16()
        report(f"rms_norm {shape}", rms_norm(x, w, 1e-6), ref(x, w, 1e-6), tol=0)   # only fp32 sum order may differ
        r = torch.randn(shape).bfloat16(); out, summed = add_rms_norm(x, r, w, 1e-6)
        report(f"add_rms_norm sum {shape}", summed, x + r)
        report(f"add_rms_norm out {shape}", out, ref(x + r, w, 1e-6), tol=0)

def swiglu():
    from kernels.swiglu import swiglu as kernel
    for shape in ((4, 2 * 1500), (2, 3, 2 * 64)):
        p = (torch.randn(shape) * 2).bfloat16(); g, u = p.chunk(2, -1)
        report(f"swiglu {shape}", kernel(p.contiguous()), F.silu(g) * u, tol=0)

def splitpath():
    """FP32 split partials consumed directly by swiglu / add_rms_norm (kernels/merged.py load_merged, SPLITS > 1)."""
    from kernels import linear as L
    from kernels.swiglu import swiglu as kernel
    from kernels.rmsnorm import add_rms_norm
    m, n, k = 6, 256, 300
    x = torch.randn(m, k).bfloat16(); w = (torch.randn(n, k) * 0.05).bfloat16(); config = L._candidates(m, n, k)[0]
    split, merged = L._project(x, w, config, split_ok=True), L._project(x, w, config)
    print("split config", config, "->", type(split).__name__, tuple(split.partial.shape), flush=True)
    report("swiglu(Split) == swiglu(merged)", kernel(split), kernel(merged))
    r = torch.randn(m, n).bfloat16(); g = torch.randn(n).bfloat16()
    a, b = add_rms_norm(split, r, g, 1e-6), add_rms_norm(merged, r, g, 1e-6)
    report("add_rms_norm(Split) out", a[0], b[0]); report("add_rms_norm(Split) sum", a[1], b[1])

def linear():
    from kernels import linear as L
    for m, n, k in ((1, 200, 300), (5, 200, 300), (16, 130, 517), (20, 70, 260), (48, 70, 260), (64, 128, 256),  # these two: 64-lane tiles
                    (16, 256, 256), (20, 512, 384)):  # whole programs of four tiles: "tmah" (no descriptor op here -> its _hoist_trans_gemm twin)
        x = torch.randn(m, k).bfloat16(); w = (torch.randn(n, k) * 0.05).bfloat16()
        exact = (x.double() @ w.double().T)
        native = F.linear(x, w)
        configs = [c for c in L._candidates(m, n, k)]
        for config in configs:
            try:
                got = L._project(x, w, config)
            except Exception as e:
                print(f"ERROR linear {config} m={m}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
            ulp = exact.abs().clamp_min(1e-30).log2().floor().exp2() / 128   # one bf16 step at that magnitude
            err = ((got.double() - exact).abs() / ulp).max()
            print(f"{'PASS' if err <= 0.75 else 'FAIL'} linear {config} m={m} n={n} k={k}: max err {float(err):.3f} bf16-ulp vs exact product "
                  f"(native F.linear: {float(((native.double() - exact).abs() / ulp).max()):.3f}); equal-to-native frac={float((got == native).float().mean()):.4f}", flush=True)
        if m > 1 and n % 64 == 0 and k % 128 == 0:
            # "tmap" (here always its twin _persist_trans_gemm: no descriptor on a CPU) must be BIT-EQUAL to
            # "trans" with one split: same operands, same K order, one rounding. Also with a 5-SM rule, so
            # programs walk several tiles (the real 132-SM rule gives these small shapes one tile each).
            from kernels import gemm
            trans = L._project(x, w, ("trans", 64, 128, 1, 4))
            report(f"tmap twin == trans(splits=1) m={m} n={n} k={k}", L._project(x, w, ("tmap", 64, 128, 1, 4)), trans)
            rule = L.persistent_programs
            L.persistent_programs = lambda tiles: gemm.persistent_programs(tiles, 5)
            try:
                report(f"tmap twin (5 programs) == trans(splits=1) m={m} n={n} k={k}", L._project(x, w, ("tmap", 64, 128, 1, 4)), trans)
            finally:
                L.persistent_programs = rule
        if m > 1:
            config = configs[0]; split = L._project(x, w, config, split_ok=True)
            if hasattr(split, "partial"):
                report(f"Split partials sum == merged {config}", split.partial.sum(0).bfloat16(), L._project(x, w, config), tol=0)

def attention():
    from kernels import decode_attention as A
    src = open("/emu/check_block_attention.py").read()
    G = {"math": math, "torch": torch}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef): exec(compile(ast.Module([node], []), "emu", "exec"), G)
    for B, T, Hq, Hkv, D, C in ((2, 4, 4, 2, 64, 90), (1, 5, 8, 2, 64, 300)):
        q = torch.randn(B, T, Hq, D).bfloat16(); k = torch.randn(B, Hkv, C, D).bfloat16(); v = torch.randn(B, Hkv, C, D).bfloat16()
        position = torch.randint(0, C - T, (B,))
        for b in range(B): k[b, :, int(position[b]) + T:] = 3e4; v[b, :, int(position[b]) + T:] = 3e4   # junk tail must not be read
        for chain in (T, 2, 1):
            chains = torch.full((B,), chain, dtype=torch.int64)
            want = G["reference"](q.double(), k.double(), v.double(), position, D ** -0.5, chain)
            for layout in (None, (32, 1, 4), (64, 3, 4)):
                A._BLOCK_LAYOUTS.clear()
                if layout: A._BLOCK_LAYOUTS[(q.device, B, T, Hq, Hkv, C, D)] = layout
                got = A.block_attention(q, k, v, position, D ** -0.5, chains)
                report(f"block_attention B{B} T{T} Hq{Hq} C{C} chain={chain} layout={layout or 'default'}", got, want.bfloat16(), tol=2e-2)
    B, Hq, Hkv, D, C = 2, 8, 2, 64, 150
    q = torch.randn(B, Hq, 1, D).bfloat16(); k = torch.randn(B, Hkv, C, D).bfloat16(); v = torch.randn_like(k)
    position = torch.tensor([97]); k[:, :, 98:] = 3e4
    want = G["reference"](q.transpose(1, 2).double(), k.double(), v.double(), position.expand(B), D ** -0.5, 1)
    for config in ((32, 4, 4), (64, 1, 4)):
        report(f"_decode_partials/_decode_merge {config}", A._attend(q, k, v, position, D ** -0.5, config), want.bfloat16(), tol=2e-2)

def qkrope():
    from kernels.qk_rope import qk_rope_cache
    class Norm:
        def __init__(self, dim): self.weight, self.variance_epsilon = torch.randn(dim).bfloat16(), 1e-6
    def norm(x, n):
        v = x.float(); v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + n.variance_epsilon)
        return n.weight * v.to(x.dtype)
    def rope(x, cos, sin):   # x [B,T,H,D], phases [T,D] or [B,T,D]
        half = x.shape[-1] // 2; rot = torch.cat((-x[..., half:], x[..., :half]), -1)
        return x * cos[..., None, :] + rot * sin[..., None, :]
    B, Hq, Hkv, D, C = 2, 4, 2, 64, 40
    qn, kn = Norm(D), Norm(D)
    for T, mode in ((1, "decode"), (5, "prefill"), (4, "rows")):
        packed = torch.randn(B, T, (Hq + 2 * Hkv) * D).bfloat16()
        rows = mode == "rows"
        angle = torch.randn((B if rows else 1), T, D); cos, sin = angle.cos().bfloat16().contiguous(), angle.sin().bfloat16().contiguous()
        position = torch.arange(T) if mode == "prefill" else torch.tensor([7, 19]) if rows else torch.tensor([11])
        keys = torch.zeros(B, Hkv, C, D).bfloat16(); values = torch.zeros_like(keys)
        q = qk_rope_cache(packed, qn, kn, cos, sin, position, keys, values, Hq, prefill=mode == "prefill", rows=rows)
        x = packed.view(B, T, Hq + 2 * Hkv, D)
        wq = rope(norm(x[:, :, :Hq], qn), cos, sin); wk = rope(norm(x[:, :, Hq:Hq + Hkv], kn), cos, sin); wv = x[:, :, Hq + Hkv:]
        rk, rv = torch.zeros_like(keys), torch.zeros_like(values)
        for b in range(B):
            start = 0 if mode == "prefill" else int(position[b if rows else 0])
            rk[b, :, start:start + T] = wk[b].transpose(0, 1); rv[b, :, start:start + T] = wv[b].transpose(0, 1)
        report(f"qk_rope {mode} query", q.transpose(1, 2).contiguous(), wq.contiguous(), tol=0)
        report(f"qk_rope {mode} key cache", keys, rk, tol=0); report(f"qk_rope {mode} value cache", values, rv)

def argmax():
    """kernels/argmax.py must be torch.argmax, including the first-index rule on ties (within and across blocks)."""
    from kernels.argmax import argmax as kernel
    for shape, block in (((3, 1000), 256), ((2, 2, 777), 64), ((5, 4096), 1024), ((1, 300), 512)):
        logits = (torch.randn(shape) * 4).bfloat16()
        report(f"argmax {shape} block={block}", kernel(logits, block=block), logits.argmax(-1))
        coarse = torch.randint(-3, 4, shape).bfloat16()          # many exact ties
        report(f"argmax ties {shape} block={block}", kernel(coarse, block=block), coarse.argmax(-1))
        flat = torch.zeros(shape).bfloat16(); flat[..., -1] = 1.0  # maximum in the last, ragged block
        report(f"argmax last {shape} block={block}", kernel(flat, block=block), flat.argmax(-1))
        negative = torch.full(shape, -7.0).bfloat16()            # all equal and below the mask's fill
        report(f"argmax equal {shape} block={block}", kernel(negative, block=block), negative.argmax(-1))

def fused_argmax():
    """kernels/argmax.py fused lm_head+argmax == torch.argmax of the BF16-rounded product (first index on ties)."""
    from kernels.argmax import fused_argmax as kernel
    for m, n, k in ((16, 640, 256), (5, 512, 384), (20, 1280, 128), (32, 256, 256)):
        x = torch.randn(m, k).bfloat16(); w = (torch.randn(n, k) * 0.05).bfloat16()
        want = (x.float() @ w.float().T).bfloat16().argmax(-1)
        report(f"fused_argmax m={m} n={n} k={k}", kernel(x, w), want)
        tied = torch.zeros(m, k).bfloat16(); tied[:, 0] = 1.0; wt = torch.zeros(n, k).bfloat16(); wt[::7, 0] = 1.0  # many exact ties
        report(f"fused_argmax ties m={m} n={n}", kernel(tied, wt), (tied.float() @ wt.float().T).bfloat16().argmax(-1))

def qkrope_table():
    """QK-RoPE reading the whole cos/sin tables at position + phase == the gathered-input path, bit for bit."""
    from kernels.qk_rope import qk_rope_cache
    class Norm:
        def __init__(self, dim): self.weight = (torch.randn(dim) * 0.1 + 1).bfloat16(); self.variance_epsilon = 1e-6
    B, T, Hq, Hkv, D, C = 2, 4, 4, 2, 64, 90
    q_norm, k_norm = Norm(D), Norm(D)
    packed = torch.randn(B, T, (Hq + 2 * Hkv) * D).bfloat16()
    table_cos = torch.randn(C, D).bfloat16(); table_sin = torch.randn(C, D).bfloat16()
    position = torch.tensor([10, 50]); phases = torch.tensor([[0, 1, 2, 1], [0, 1, 1, 1]])
    gathered = position[:, None] + phases
    outs, caches = [], []
    for table in (False, True):
        keys = torch.zeros(B, Hkv, C, D).bfloat16(); values = torch.zeros(B, Hkv, C, D).bfloat16()
        if table:
            q = qk_rope_cache(packed, q_norm, k_norm, table_cos, table_sin, position, keys, values, Hq, rows=True, phases=phases)
        else:
            q = qk_rope_cache(packed, q_norm, k_norm, table_cos[gathered].contiguous(), table_sin[gathered].contiguous(), position, keys, values, Hq, rows=True)
        outs.append(q.clone()); caches.append((keys.clone(), values.clone()))
    report("qk_rope table mode == gathered mode (query)", outs[1], outs[0])
    report("qk_rope table mode == gathered mode (keys)", caches[1][0], caches[0][0])
    report("qk_rope table mode == gathered mode (values)", caches[1][1], caches[0][1])

def embed_norm():
    """Embedding gather + first norm in one kernel == gather then rms_norm, bit for bit."""
    from kernels.rmsnorm import rms_norm, embed_rms_norm
    table = (torch.randn(50, 256) * 0.05).bfloat16(); w = (torch.randn(256) * 0.1 + 1).bfloat16()
    ids = torch.tensor([[3, 49, 0, 7], [12, 12, 1, 30]])
    out, hidden = embed_rms_norm(ids, table, w, 1e-6)
    report("embed_rms_norm hidden == table[ids]", hidden, table[ids])
    report("embed_rms_norm normalized == rms_norm(table[ids])", out, rms_norm(table[ids].contiguous(), w, 1e-6))

which = sys.argv[1:] or ["embed_norm", "qkrope_table", "fused_argmax", "argmax", "qkrope", "splitpath", "rmsnorm", "swiglu", "linear", "attention"]
for name in which: section(globals()[name])
