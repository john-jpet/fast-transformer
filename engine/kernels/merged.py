"""A projection's FP32 split partials, consumed without a separate merge launch.

The split GEMM leaves SPLITS FP32 partial sums per output element. The merge
kernel used to add them and round to BF16; a consumer kernel can do exactly
that at load time instead: sum the partials, round to BF16, carry on. Same
arithmetic, one launch fewer per projection.
"""

import triton
import triton.language as tl


class Split:
    """FP32 partials [S, M, N] standing for a BF16 tensor of logical ``shape`` (..., N)."""

    __slots__ = ("partial", "shape")

    def __init__(self, partial, shape):
        self.partial, self.shape = partial, tuple(shape)

    @property
    def splits(self):
        return self.partial.shape[0]

    @property
    def count(self):
        return self.partial.shape[1] * self.partial.shape[2]


def source(x):
    """(tensor to read, SPLITS, elements per split, logical shape) for a tensor or a Split."""
    if isinstance(x, Split):
        return x.partial, x.splits, x.count, x.shape
    return x, 1, 1, tuple(x.shape)


@triton.jit
def load_merged(ptr, offsets, mask, COUNT: tl.constexpr, SPLITS: tl.constexpr):
    """BF16 values at ``offsets``: stored directly, or the rounded sum of the FP32 partials."""
    if SPLITS == 1:
        return tl.load(ptr + offsets, mask, other=0)
    total = tl.load(ptr + offsets, mask, other=0)
    for split in tl.static_range(1, SPLITS):
        total += tl.load(ptr + split * COUNT + offsets, mask, other=0)
    return total.to(tl.bfloat16)
