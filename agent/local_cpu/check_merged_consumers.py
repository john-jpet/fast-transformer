"""CPU index/cast checks for E03 merged projection consumers; no GPU claims."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
import torch
from kernels.merged import Split, source


def loaded(partials, offsets, mask):
    count = partials.shape[1] * partials.shape[2]
    flat = partials.flatten()
    result = torch.zeros_like(offsets, dtype=torch.float32)
    result[mask] = flat[offsets[mask]]
    for split in range(1, partials.shape[0]):
        result[mask] += flat[(split * count + offsets)[mask]]
    return result.bfloat16()


torch.set_num_threads(1)
torch.manual_seed(17)
cases = 0
for batch, tokens in ((1, 16), (4, 4), (3, 5), (16, 2)):
    rows = batch * tokens
    for splits in (2, 4, 8):
        for width in (2560, 6144):
            # Dyadic partials give exact FP32 sums, isolating indexing/casts
            # from allowed differences between FP32 reduction trees.
            partials = torch.randint(-1000, 1001, (splits, rows, width)).float() / 512
            logical = (batch, tokens, width)
            src, ns, count, shape = source(Split(partials, logical))
            assert src is partials and ns == splits and count == rows * width and shape == logical
            expected = partials.sum(0).bfloat16()
            if width == 2560:
                residual = torch.randn(rows, width).bfloat16()
                for row in range(rows):
                    cols = torch.arange(4096)
                    mask = cols < width
                    x = loaded(partials, row * width + cols, mask)[:width]
                    assert torch.equal(x, expected[row])
                    summed = (x.float() + residual[row].float()).bfloat16()
                    assert torch.equal(summed, expected[row] + residual[row])
            else:
                # Both Q/K halves and V loads use the same packed row stride.
                for row in range(rows):
                    for head in range(40):
                        cols = torch.arange(128)
                        offsets = row * width + head * 128 + cols
                        mask = cols < 128
                        got = loaded(partials, offsets, mask)
                        paired = loaded(partials, row * width + head * 128 + (cols + 64) % 128, mask)
                        assert torch.equal(got, expected[row, head * 128:(head + 1) * 128])
                        assert torch.equal(paired, torch.roll(got, 64))
                        if head >= 32:
                            offsets = row * width + (40 + head - 32) * 128 + cols
                            value = loaded(partials, offsets, mask)
                            assert torch.equal(value, expected[row, (head + 8) * 128:(head + 9) * 128])
            cases += 1
print(f"merged consumer indexing and BF16 projection/residual boundaries passed: {cases} cases")
