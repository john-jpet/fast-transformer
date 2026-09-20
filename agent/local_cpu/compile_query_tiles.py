"""Offline H100 compilation of both query tiles, including the TMA path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials, _block_partials_tma

for tile in (32, 16):
    for splits in (1, 4):
        constants = dict(TOKENS=16, GROUPS=4, Q_HEADS=32, KV_HEADS=8,
                         DIM=128, CAPACITY=560, SPLITS=splits, CHUNK=(560 + splits - 1) // splits,
                         SCALE=128 ** -0.5, BLOCK_M=tile, BLOCK_N=64)
        signature = dict(q_ptr="*bf16", k_ptr="*bf16", v_ptr="*bf16", position_ptr="*i64",
                         chain_ptr="*i64", partial_ptr="*fp32", stats_ptr="*fp32", out_ptr="*bf16")
        for prefix in (False, True):
            result = compile_kernel(_block_partials, signature, dict(constants, PREFIX=prefix), num_warps=4, num_stages=2)
            print("tile", tile, "splits", splits, "prefix", prefix, "shared", result.metadata.shared)
        signature.update(kd_ptr="*i8", vd_ptr="*i8")
        compile_kernel(_block_partials_tma, signature, dict(constants, LIMIT=512, TMA=True), num_warps=4, num_stages=2)
