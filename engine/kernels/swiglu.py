"""SwiGLU from packed gate/up projections, preserving the SiLU BF16 output."""

import torch
import triton
import triton.language as tl

from kernels.merged import load_merged, source
from kernels.tune import pick


@triton.jit
def _swiglu(packed, output, WIDTH: tl.constexpr, BLOCK: tl.constexpr, COUNT: tl.constexpr = 1, SPLITS: tl.constexpr = 1):
    # One program per (row, column block): no per-element division.
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = column < WIDTH
    gate = load_merged(packed, row * (2 * WIDTH) + column, valid, COUNT, SPLITS).to(tl.float32)
    up = load_merged(packed, row * (2 * WIDTH) + WIDTH + column, valid, COUNT, SPLITS).to(tl.float32)
    # Native SiLU returns BF16 before the separate gate/up multiplication.
    activated = (gate / (1.0 + tl.exp(-gate))).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + row * WIDTH + column, activated * up, valid)


def swiglu(packed):
    """Contiguous BF16 [..., 2*I] gate/up input; new contiguous [..., I] output."""
    packed, splits, count, shape = source(packed)
    width = shape[-1] // 2
    output = torch.empty((*shape[:-1], width), device=packed.device, dtype=torch.bfloat16)
    rows = output.numel() // width

    def launch(warps):
        _swiglu[(rows, triton.cdiv(width, 1024))](
            packed, output, WIDTH=width, BLOCK=1024, COUNT=count, SPLITS=splits, num_warps=warps,
        )

    launch(pick(("swiglu", rows, width), 4, (1, 2, 8), launch) if rows <= 16 else 4)
    return output
