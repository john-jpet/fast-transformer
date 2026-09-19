"""Verify real candidate split coverage and compile the newly offered shapes."""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
from kernels.linear import _candidates, _skinny_gemm
from offline_compile import compile_kernel

checks = 0
for m in (5, 15, 16, 24, 32):
    for n, k in ((6144, 2560), (19456, 2560), (2560, 9728), (2560, 4096), (151936, 2560)):
        configs = _candidates(m, n, k)
        base = configs[0]
        # Existing configuration remains available exactly as before.
        assert base[0:3] == ("gemm", 64, 128)
        for kind, bn, bk, splits, warps in configs:
            chunk = math.ceil(k / (splits * bk)) * bk
            seen = [0] * k
            for split in range(splits):
                for start in range(split * chunk, (split + 1) * chunk, bk):
                    for index in range(start, min(start + bk, k)):
                        seen[index] += 1
            assert seen == [1] * k
            checks += 1
        if base[3] == 8 and k == 2560:
            assert ("gemm", 64, 128, 5, 4) in configs
for m, bm, n, k, splits in ((16, 16, 6144, 2560, 5), (32, 32, 6144, 2560, 5),
                           (16, 16, 2560, 9728, 4), (32, 32, 2560, 9728, 4)):
    compile_kernel(
        _skinny_gemm, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"},
        {"M": m, "N": n, "K": k, "SPLITS": splits, "CHUNK": k // splits,
         "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": bm}, num_warps=4, num_stages=2,
    )
print(f"even split candidates: {checks} coverage checks, four SM90 specializations passed")
