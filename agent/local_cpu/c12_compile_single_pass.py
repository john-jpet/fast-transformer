from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials
ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "chain_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}
for tokens, splits, block_n, bm in ((16, 1, 64, 64), (4, 1, 128, 16), (4, 8, 64, 16), (2, 2, 64, 16)):
    sig = dict(ptrs)
    if splits == 1: sig["partial_ptr"] = sig["stats_ptr"] = "*bf16"
    compile_kernel(_block_partials, sig, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 2084, "SPLITS": splits, "CHUNK": -(-2084 // splits), "SCALE": 128 ** -0.5, "BLOCK_M": bm, "BLOCK_N": block_n}, num_warps=4, num_stages=2)
print("single-pass and split block attention compile for cuda:90")
