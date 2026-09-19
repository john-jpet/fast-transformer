"""SM90 compile coverage for E02 split and single-pass tree attention."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials, _block_merge

for tokens, capacity, splits, block_n in (
    (16, 560, 1, 64), (16, 560, 9, 128), (16, 560, 18, 32),
    (8, 560, 1, 64), (5, 1060, 4, 128), (4, 2084, 1, 64),
    (4, 2084, 4, 128), (3, 550, 1, 64), (2, 642, 1, 64),
    (2, 642, 2, 64), (2, 4162, 1, 128),
):
    ptrs = {
        "q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16",
        "position_ptr": "*i64", "chain_ptr": "*i64",
        "partial_ptr": "*bf16" if splits == 1 else "*fp32",
        "stats_ptr": "*bf16" if splits == 1 else "*fp32", "out_ptr": "*bf16",
    }
    compile_kernel(
        _block_partials, ptrs,
        {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8,
         "DIM": 128, "CAPACITY": capacity, "SPLITS": splits,
         "CHUNK": (capacity + splits - 1) // splits, "SCALE": 128 ** -0.5,
         "BLOCK_M": max(16, 1 << (tokens * 4 - 1).bit_length()), "BLOCK_N": block_n},
        num_warps=4, num_stages=2,
    )
    if splits > 1:
        compile_kernel(
            _block_merge, {"partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"},
            {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8,
             "DIM": 128, "SPLITS": splits, "BLOCK_S": 1 << (splits - 1).bit_length()},
            num_warps=4,
        )
print("tree verification single-pass and split kernels compile for cuda:90")
