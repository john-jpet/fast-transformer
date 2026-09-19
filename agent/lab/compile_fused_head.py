from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root/'engine'), str(root/'agent/local_cpu')]
from offline_compile import compile_kernel
from fused_head_prototype import _head_tiles
for m, n in ((16,151936),(32,151936),(5,151939)):
    out = compile_kernel(_head_tiles,
        dict(x_ptr='*bf16',weight_ptr='*bf16',best_value='*fp32',best_index='*i64'),
        dict(M=m,N=n,K=2560,SPLITS=1,CHUNK=2560,BLOCK_N=64,BLOCK_K=128,
             BLOCK_M=16 if m<=16 else 32,EVEN_M=m in (16,32),EVEN_N=n%64==0,EVEN_K=True,WIDE=False),
        num_warps=4,num_stages=2)
    print(m,n,'shared',out.metadata.shared,'PTX local declarations',out.asm['ptx'].count('.local'))
