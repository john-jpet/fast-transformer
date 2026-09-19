from offline_compile import compile_kernel
from kernels.rmsnorm import _add_rms_norm_kernel
from kernels.swiglu import _swiglu
from kernels.qk_rope import _qk_rope_cache
for splits, dtype in ((1, "*bf16"), (8, "*fp32"), (2, "*fp32"), (5, "*fp32"), (4, "*fp32")):
    compile_kernel(_add_rms_norm_kernel, {"x_ptr": dtype, "residual_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "sum_ptr": "*bf16", "COUNT": "i32"}, {"WIDTH": 2560, "EPS": 1e-6, "BLOCK": 4096, "SPLITS": splits}, num_warps=4)
    compile_kernel(_swiglu, {"packed": dtype, "output": "*bf16", "COUNT": "i32"}, {"WIDTH": 9728, "BLOCK": 1024, "SPLITS": splits}, num_warps=4)
    sig = {"packed": dtype, "q_weight": "*bf16", "k_weight": "*bf16", "cos": "*bf16", "sin": "*bf16", "position": "*i64", "query": "*bf16", "keys": "*bf16", "values": "*bf16", "COUNT": "i32"}
    for table in (False, True):
        compile_kernel(_qk_rope_cache, {**sig, "phases": "*i64"}, {"Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 549, "Q_EPS": 1e-6, "K_EPS": 1e-6, "TOKENS": 4, "PREFILL": False, "BLOCK": 128, "ROWS": True, "SPLITS": splits, "TABLE": table}, num_warps=4)
print("merge-fused consumer kernels compile for cuda:90")
