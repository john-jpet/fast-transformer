"""Compile all E01 kernel layouts for SM90; no GPU or model weights needed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from offline_compile import compile_kernel
from kernels.gated_linear import _paired_projection, _merge_activate

for m, bm, bn, bk, warps in (
    (1, 16, 32, 64, 4), (16, 16, 32, 64, 4), (32, 32, 32, 64, 4),
    (8192, 32, 64, 32, 4), (8192, 64, 64, 32, 8), (65, 32, 64, 32, 4),
):
    kernel = compile_kernel(
        _paired_projection, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"},
        {"M": m, "I": 9728, "K": 2560, "BM": bm, "BN": bn, "BK": bk},
        num_warps=warps, num_stages=3,
    )
    print(f"paired M={m} tile={bm}x{bn}x{bk} shared={kernel.metadata.shared}")
    # Inspect generated PTX for the explicit round-to-BF16 boundaries.
    assert "bf16" in kernel.asm["ptx"]

for m, splits in ((1, 2), (16, 2), (16, 5), (16, 8), (32, 4)):
    compile_kernel(
        _merge_activate, {"partial_ptr": "*fp32", "out_ptr": "*bf16"},
        {"M": m, "I": 9728, "SPLITS": splits, "BS": 1 << (splits - 1).bit_length(), "BLOCK": 512},
        num_warps=4,
    )
print("gated projection and merge epilogues compile for cuda:90")
