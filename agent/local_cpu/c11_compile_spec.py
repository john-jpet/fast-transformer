# _block_partials (single-pass signature) is compiled by compile_single_pass.py.
from offline_compile import compile_kernel
from kernels.spec import _propose, _settle, _relocate
for size, T, D in ((551, 16, (5, 8, 13, 14)), (2086, 4, (1, 2, 3, 3)), (70, 2, (1, 1, 1, 1)), (551, 8, (2, 4, 6, 7))):
    block = 1 << (size - 1).bit_length(); lanes = T - 1 - min(D)
    compile_kernel(_propose, {"history": "*i64", "position": "*i64", "successor": "*i64", "stale": "*i64", "tokens": "*i64", "chains": "*i64", "phases": "*i64"}, {"SIZE": size, "TOKENS": T, "MAXLEN": 8, "D0": D[0], "D1": D[1], "D2": D[2], "D3": D[3], "LANES": lanes, "ALTERNATES": min(lanes, 3), "TOP": 8, "BLOCK": block, "BLOCK_S": max(1, 1 << (max(lanes, 1) - 1).bit_length()), "BLOCK_T": 1 << (T - 1).bit_length()}, num_warps=4)
    compile_kernel(_settle, {"tokens": "*i64", "greedy": "*i64", "position": "*i64", "limit": "*i64", "history": "*i64", "result": "*i64", "move_from": "*i64", "move_to": "*i64", "chains": "*i64", "stale": "*i64"}, {"SIZE": size, "TOKENS": T, "BLOCK": 1 << (T - 1).bit_length()}, num_warps=1)
compile_kernel(_relocate, {"store": "*bf16", "move_from": "*i64", "move_to": "*i64"}, {"BATCH": 2, "KV_HEADS": 8, "CAPACITY": 553, "DIM": 128, "BLOCK_H": 8}, num_warps=1)
print("speculation kernels (per-row chains) compile for cuda:90")
