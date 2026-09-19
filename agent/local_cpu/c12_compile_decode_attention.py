from offline_compile import compile_kernel
from kernels.decode_attention import _decode_partials
for capacity, block_n, splits, chunk in ((545, 64, 1, 576), (2081, 64, 4, 576), (641, 32, 4, 192)):
    compile_kernel(
        _decode_partials,
        {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32"},
        {"GROUPS": 4, "DIM": 128, "CAPACITY": capacity, "SPLITS": splits, "CHUNK": chunk, "SCALE": 128 ** -0.5, "BLOCK_M": 16, "BLOCK_N": block_n},
        num_warps=4, num_stages=2,
    )
print("plain decode attention compiles for cuda:90")
