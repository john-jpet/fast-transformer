"""Execute query tiling against the original full-query layout and dense reference."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "interp"), str(HERE.parents[1] / "engine"), str(HERE)]
import interp_bf16
import numpy2_memory
import torch
from kernels import decode_attention as attention
from check_block_attention import reference

torch.manual_seed(391)
torch.set_num_threads(1)
cases = 0
for batch, tokens, capacity in ((1, 16, 79), (2, 8, 91), (2, 5, 67), (1, 3, 33), (2, 2, 39)):
    heads, kv, dim = 8, 2, 128
    query = torch.randn(batch, tokens, heads, dim).bfloat16()
    key = torch.randn(batch, kv, capacity, dim).bfloat16()
    value = torch.randn_like(key)
    position = torch.tensor([capacity - tokens if b == 0 else 0 for b in range(batch)])
    for b in range(batch):
        key[b, :, position[b] + tokens:] = float("nan")
        value[b, :, position[b] + tokens:] = float("nan")
    for chain_length in (1, min(3, tokens), tokens):
        chain = torch.full((batch,), chain_length, dtype=torch.int64)
        expected = reference(query.double(), key.double(), value.double(), position, dim ** -0.5, chain_length).bfloat16()
        for layout in ((32, 1, 4), (32, 3, 4)):
            attention.QUERY_TILE = 64
            original = attention._launch_block("plain", query, key, value, position, chain, dim ** -0.5, layout)
            for tile in (32, 16):
                attention.QUERY_TILE = tile
                actual = attention._launch_block("plain", query, key, value, position, chain, dim ** -0.5, layout)
                if not torch.equal(actual, original):
                    difference = (actual.float() - original.float()).abs()
                    print("tile comparison", batch, tokens, layout, tile, "max error", difference.max().item(), "different", (actual != original).sum().item(), flush=True)
                assert torch.allclose(actual.float(), original.float(), atol=0.002, rtol=0.008), (batch, tokens, layout, tile, "full-layout mismatch")
                assert torch.isfinite(actual).all(), "read poisoned unused cache"
                assert (actual.float() - expected.float()).abs().max() <= 0.02
                cases += 1
print(f"PASS: {cases} tiled attention cases; full-query layout within BF16 tolerance; dense reference within 0.02")
