"""Unsubmitted prototype: complete BF16 vocabulary GEMM tiles -> tile winners."""
from pathlib import Path
root = Path(__file__).resolve().parents[2]
s = (root / 'engine/kernels/gemm.py').read_text()
s = s[s.index('@triton.jit\ndef _exact_gemm'):s.index('@triton.jit\ndef _trans_gemm')]
s = s.replace('_exact_gemm', '_head_tiles').replace('x_ptr, weight_ptr, out_ptr,', 'x_ptr, weight_ptr, best_value, best_index,')
s = s.replace('    rows = tl.arange', '    tl.static_assert(SPLITS == 1)\n    rows = tl.arange', 1)
s = s[:s.index('    _store_tile(')] + '''    # Match materialized BF16 logits BEFORE selecting a winner.
    logits = tl.where(col_ok, acc.to(tl.bfloat16).to(tl.float32), -float("inf"))
    maximum, local = tl.max(logits, axis=1, return_indices=True, return_indices_tie_break_left=True)
    tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    tl.store(best_value + rows * tiles + tile, maximum, rows < M)
    tl.store(best_index + rows * tiles + tile, tile * BLOCK_N + local, rows < M)
'''
(root / 'agent/lab/fused_head_prototype.py').write_text('import triton\nimport triton.language as tl\nfrom kernels.gemm import _load_tile\n\n' + s)
print('Built unsplit LM-head tile-winner prototype; not imported by engine')
