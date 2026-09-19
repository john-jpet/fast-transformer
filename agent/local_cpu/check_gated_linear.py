"""CPU emulation of paired projection indexing and the fused merge epilogue.

Does not establish GPU numerical parity or performance. Run with CPU torch.
"""

import math
import torch
from torch.nn import functional as F


def check_tiles(m, width, k, bm, bn, bk):
    tiles_m = math.ceil(m / bm)
    tiles_n = math.ceil(width / bn)
    seen = set()
    for pid in range(tiles_m * tiles_n):
        group = pid // (8 * tiles_n)
        first_m = group * 8
        group_m = min(tiles_m - first_m, 8)
        within = pid % (8 * tiles_n)
        tile_m = first_m + within % group_m
        tile_n = within // group_m
        assert 0 <= tile_m < tiles_m and 0 <= tile_n < tiles_n
        assert (tile_m, tile_n) not in seen
        seen.add((tile_m, tile_n))
    assert len(seen) == tiles_m * tiles_n
    # Evaluate the same masked address formulas for an odd small tensor.
    if m * width * k > 1_000_000:
        return
    x = torch.randn(m, k).bfloat16()
    weight = torch.randn(2 * width, k).bfloat16()
    output = torch.full((m, 2 * width), float("nan"))
    for tile_m, tile_n in seen:
        rows = torch.arange(tile_m * bm, min((tile_m + 1) * bm, m))
        columns = torch.arange(tile_n * bn, min((tile_n + 1) * bn, width))
        gate = torch.zeros(len(rows), len(columns))
        up = torch.zeros_like(gate)
        for start in range(math.ceil(k / bk)):
            reduction = torch.arange(start * bk, min((start + 1) * bk, k))
            xx = x.flatten()[rows[:, None] * k + reduction[None, :]].float()
            wg = weight.flatten()[columns[None, :] * k + reduction[:, None]].float()
            wu = weight.flatten()[(width + columns[None, :]) * k + reduction[:, None]].float()
            gate += xx @ wg
            up += xx @ wu
        output[rows[:, None], columns[None, :]] = gate
        output[rows[:, None], width + columns[None, :]] = up
    torch.testing.assert_close(output, x.float() @ weight.float().T, rtol=1e-5, atol=1e-5)


def activate(gate, up):
    gate = gate.bfloat16().float()
    up = up.bfloat16().float()
    silu = (gate / (1.0 + torch.exp(-gate))).bfloat16().float()
    return (silu * up).bfloat16()


def check_merge(m, width, splits, block=512):
    partials = torch.randn(splits, m, 2 * width)
    flat = partials.flatten()
    output = torch.full((m * width,), float("nan"), dtype=torch.bfloat16)
    bs = 1 << (splits - 1).bit_length()
    for pid in range(math.ceil(m * width / block)):
        offsets = pid * block + torch.arange(block)
        row, column = offsets // width, offsets % width
        ss = torch.arange(bs)
        address = ss[:, None] * m * (2 * width) + row[None, :] * (2 * width) + column[None, :]
        mask = (ss[:, None] < splits) & (offsets[None, :] < m * width)
        gate, up = torch.zeros(bs, block), torch.zeros(bs, block)
        gate[mask] = flat[address[mask]]
        up[mask] = flat[(address + width)[mask]]
        actual = activate(gate.sum(0), up.sum(0))
        valid = offsets < m * width
        output[offsets[valid]] = actual[valid]
    summed = partials.sum(0)
    expected = activate(summed[:, :width], summed[:, width:]).flatten()
    # Both are FP32 reductions; differing CPU sum trees can cross BF16 ties.
    torch.testing.assert_close(output.float(), expected.float(), rtol=0.02, atol=0.001)
    # Native's activation has the same rounding stages.
    rounded = summed.bfloat16()
    native = (F.silu(rounded[:, :width]) * rounded[:, width:]).flatten()
    torch.testing.assert_close(output.float(), native.float(), rtol=0.02, atol=0.001)
    assert torch.isfinite(output).all()


def main():
    torch.manual_seed(2718)
    torch.set_num_threads(1)
    cases = 0
    for m in (1, 2, 16, 31, 32, 65, 255, 257, 513, 8192):
        for bm, bn, bk in ((16, 32, 64), (32, 64, 32), (64, 64, 32)):
            check_tiles(m, 67, 37, bm, bn, bk)
            check_tiles(m, 9728, 2560, bm, bn, bk)
            cases += 2
    for m in (1, 3, 16, 32):
        for width in (1, 7, 513, 9728):
            for splits in (1, 2, 3, 8):
                check_merge(m, width, splits)
                cases += 1
    gate, up = torch.randn(100_000) * 3, torch.randn(100_000)
    expected = activate(gate, up)
    wrong = (F.silu(gate) * up).bfloat16()
    assert (expected != wrong).float().mean() > 0.1, "rounding test must reject a FP32 reformulation"
    print(f"gated projection CPU emulation passed: {cases} cases and BF16 rounding sensitivity")


if __name__ == "__main__":
    main()
