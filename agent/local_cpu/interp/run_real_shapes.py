"""Interpreter cost at true Qwen3-4B shapes (hidden 2560, MLP 9728, 32/8 heads, dim 128)."""
import interp_bf16, time, torch
from torch.nn import functional as F
from kernels import linear as L, decode_attention as A
from kernels.swiglu import swiglu
torch.manual_seed(0)
def timed(name, fn):
    t = time.time(); out = fn(); print(f"{name}: {time.time() - t:.1f}s", flush=True); return out
x = torch.randn(16, 2560).bfloat16(); w = (torch.randn(2 * 9728, 2560) * 0.02).bfloat16()
for config in L._candidates(16, w.shape[0], 2560):
    got = timed(f"gate_up projection 16x2560 -> 19456 {config}", lambda: L._project(x, w, config))
    native = F.linear(x, w); print("   equal-to-native frac", float((got == native).float().mean()), "max|diff|", float((got.float() - native.float()).abs().max()), flush=True)
timed("swiglu 16x19456", lambda: swiglu(got))
B, T, Hq, Hkv, D, C = 4, 8, 32, 8, 128, 2048 + 160
q = torch.randn(B, T, Hq, D).bfloat16(); k = torch.randn(B, Hkv, C, D).bfloat16(); v = torch.randn_like(k)
position = torch.tensor([2050, 2060, 2100, 2049]); chain = torch.tensor([8, 5, 3, 1])
got = timed(f"block_attention B{B} T{T} C{C}", lambda: A.block_attention(q, k, v, position, D ** -0.5, chain))
want = torch.zeros_like(got, dtype=torch.float64)
for b in range(B):
    for t in range(T):
        p = int(position[b]); slots = list(range(p + t + 1)) if t < int(chain[b]) else list(range(p + 1)) + [p + t]
        kk = k[b, :, slots].double().repeat_interleave(Hq // Hkv, 0); vv = v[b, :, slots].double().repeat_interleave(Hq // Hkv, 0)
        want[b, t] = (torch.softmax((kk @ q[b, t].double()[:, :, None]).squeeze(-1) * D ** -0.5, -1)[:, None, :] @ vv).squeeze(1)
print("block_attention vs float64 dense reference: max|diff|", float((got.double() - want).abs().max()), flush=True)
