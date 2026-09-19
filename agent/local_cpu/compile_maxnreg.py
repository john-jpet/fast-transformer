import re
from offline_compile import compile_kernel
from kernels.linear import _skinny_gemm
base = {"M": 16, "N": 19456, "K": 2560, "SPLITS": 2, "CHUNK": 1280, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": 16}
for opts in ({"num_warps": 4}, {"num_warps": 4, "maxnreg": 1024}, {"num_warps": 8, "maxnreg": 1024}, {"num_warps": 16, "maxnreg": 512}, {"num_warps": 12}):
    try:
        out = compile_kernel(_skinny_gemm, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"}, base, num_stages=2, **opts)
        print(opts, "ok; ptx bytes", len(out.asm["ptx"]))
    except Exception as e:
        print(opts, "FAILED", repr(e)[:200])
