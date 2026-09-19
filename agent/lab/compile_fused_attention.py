from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_cpu'))
from offline_compile import compile_kernel
from fused_attention_prototype import _fused_block
for tokens in (2, 4, 8, 16):
    ptrs = {name: '*bf16' for name in ('packed','q_gain','k_gain','cos','sin','k_ptr','v_ptr','partial_ptr','stats_ptr','out_ptr')}
    ptrs.update(position_ptr='*i64', chain_ptr='*i64')
    kernel = compile_kernel(_fused_block, ptrs,
        dict(TOKENS=tokens,GROUPS=4,Q_HEADS=32,KV_HEADS=8,DIM=128,
             Q_EPS=1e-6,K_EPS=1e-6,CAPACITY=2100,SPLITS=1,CHUNK=2100,
             SCALE=128**-0.5,BLOCK_M=max(16,tokens*4),BLOCK_N=64),
        num_warps=4,num_stages=2)
    print(tokens, kernel.metadata, 'local_declarations', kernel.asm['ptx'].count('.local'))
