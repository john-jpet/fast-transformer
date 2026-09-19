"""Compile E03 consumer variants against Triton 3.1 SM90, without CUDA."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from offline_compile import compile_kernel
from kernels.rmsnorm import _add_rms_norm_kernel
from kernels.qk_rope import _qk_rope_cache

for splits, dtype in ((1, "*bf16"), (2, "*fp32"), (4, "*fp32"), (5, "*fp32"), (8, "*fp32")):
    for warps in (1, 4, 8):
        compile_kernel(
            _add_rms_norm_kernel,
            {"x_ptr": dtype, "residual_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "sum_ptr": "*bf16"},
            {"WIDTH": 2560, "EPS": 1e-6, "BLOCK": 4096,
             "COUNT": 16 * 2560 if splits > 1 else 1, "SPLITS": splits}, num_warps=warps,
        )
    compile_kernel(
        _qk_rope_cache,
        {"packed": dtype, "q_weight": "*bf16", "k_weight": "*bf16", "cos": "*bf16", "sin": "*bf16",
         "position": "*i64", "query": "*bf16", "keys": "*bf16", "values": "*bf16"},
        {"Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 549,
         "Q_EPS": 1e-6, "K_EPS": 1e-6, "TOKENS": 4, "PREFILL": False,
         "BLOCK": 128, "ROWS": True, "COUNT": 16 * 6144 if splits > 1 else 1, "SPLITS": splits},
        num_warps=4,
    )
print("E03 residual norm and QK consumers compile for cuda:90")
