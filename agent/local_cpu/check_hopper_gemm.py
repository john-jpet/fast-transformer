"""E06 actual transposed pointer/store emulation and SM90 codegen checks."""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
import torch
from kernels.linear import _hopper_gemm
from offline_compile import compile_kernel

torch.set_num_threads(1)
torch.manual_seed(6)
for m, n, k, splits in ((5, 67, 37, 3), (16, 129, 256, 2), (31, 65, 257, 5), (32, 128, 192, 3)):
    bm, bn, bk = (16 if m <= 16 else 32), 64, 32
    chunk = math.ceil(k / (splits * bk)) * bk
    x = torch.randn(m, k, dtype=torch.float64)
    weight = torch.randn(n, k, dtype=torch.float64)
    out = torch.full((splits * m * n,), float("nan"), dtype=torch.float64)
    counts = torch.zeros_like(out, dtype=torch.int64)
    rows = torch.arange(bm)
    for tile in range(math.ceil(n / bn)):
        columns = tile * bn + torch.arange(bn)
        for split in range(splits):
            acc = torch.zeros(bn, bm, dtype=torch.float64)
            for start in range(split * chunk, (split + 1) * chunk, bk):
                reduction = start + torch.arange(bk)
                wm = (columns[:, None] < n) & (reduction[None, :] < k)
                xm = (rows[None, :] < m) & (reduction[:, None] < k)
                wa = columns[:, None] * k + reduction[None, :]
                xa = rows[None, :] * k + reduction[:, None]
                wtile, xtile = torch.zeros(bn, bk, dtype=torch.float64), torch.zeros(bk, bm, dtype=torch.float64)
                wtile[wm] = weight.flatten()[wa[wm]]
                xtile[xm] = x.flatten()[xa[xm]]
                acc += wtile @ xtile
            address = split * m * n + rows[None, :] * n + columns[:, None]
            mask = (rows[None, :] < m) & (columns[:, None] < n)
            out[address[mask]] = acc[mask]
            counts[address[mask]] += 1
    assert (counts == 1).all()
    torch.testing.assert_close(out.reshape(splits, m, n).sum(0), x @ weight.T, rtol=1e-12, atol=1e-12)

for m, bm in ((16, 16), (32, 32)):
    for n, k, splits in ((6144, 2560, 5), (19456, 2560, 2), (2560, 9728, 4), (151936, 2560, 1)):
        chunk = math.ceil(k / (splits * 128)) * 128
        kernel = compile_kernel(
            _hopper_gemm, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16" if splits == 1 else "*fp32"},
            {"M": m, "N": n, "K": k, "SPLITS": splits, "CHUNK": chunk,
             "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": bm}, num_warps=4, num_stages=2,
        )
        assert "wgmma.mma_async" in kernel.asm["ptx"], "candidate must actually use WGMMA"
print("E06 transposed pointer/store reference and eight SM90 WGMMA variants passed")
