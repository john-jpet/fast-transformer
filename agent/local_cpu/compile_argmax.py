from offline_compile import compile_kernel
from kernels.argmax import _block_argmax, _first_best, _fused_block_argmax
compile_kernel(_block_argmax, {"logits": "*bf16", "best_value": "*fp32", "best_index": "*i64"}, {"VOCAB": 151936, "BLOCKS": 19, "BLOCK": 8192}, num_warps=4)
compile_kernel(_first_best, {"best_value": "*fp32", "best_index": "*i64", "out": "*i64"}, {"BLOCKS": 19, "BLOCK_B": 32}, num_warps=1)
for m, block_m in ((16, 16), (32, 32), (5, 16)):
    compile_kernel(_fused_block_argmax, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "best_value": "*fp32", "best_index": "*i64"}, {"M": m, "N": 151936, "K": 2560, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": block_m, "EVEN_M": m == block_m}, num_warps=4, num_stages=2)
compile_kernel(_first_best, {"best_value": "*fp32", "best_index": "*i64", "out": "*i64"}, {"BLOCKS": 2374, "BLOCK_B": 4096}, num_warps=4)
print("argmax kernels compile for cuda:90")
