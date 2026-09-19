"""E07 two-stage vocabulary reduction: exact indices and SM90 compilation."""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
import torch
from kernels.argmax import _block_argmax, _first_best
from offline_compile import compile_kernel

torch.set_num_threads(1)
torch.manual_seed(7)
cases = 0
for vocab in (1, 8191, 8192, 8193, 151936):
    for shape in ((1, vocab), (4, 4, vocab)):
        x = torch.randn(shape, dtype=torch.bfloat16)
        flat = x.reshape(-1, vocab)
        variants = [x, -x.abs(), torch.full_like(x, -float('inf')), torch.zeros_like(x)]
        final = x.clone()
        final[..., -1] = float('inf')
        variants.append(final)
        tied = x.clone()
        tied[..., 0] = tied[..., -1] = 32
        variants.append(tied)
        for values in variants:
            flat = values.reshape(-1, vocab)
            blocks = math.ceil(vocab / 8192)
            padded = torch.full((flat.shape[0], blocks * 8192), -float('inf'))
            padded[:, :vocab] = flat.float()
            local_value, local_index = padded.reshape(-1, blocks, 8192).max(-1)
            local_index += torch.arange(blocks) * 8192
            chosen = local_value.argmax(-1)
            actual = local_index.gather(1, chosen[:, None]).reshape(values.shape[:-1])
            assert torch.equal(actual, values.argmax(-1))
            cases += 1
for vocab in (1, 8193, 151936):
    blocks = math.ceil(vocab / 8192)
    compile_kernel(_block_argmax,
        {'logits': '*bf16', 'best_value': '*fp32', 'best_index': '*i64'},
        {'VOCAB': vocab, 'BLOCKS': blocks, 'BLOCK': 8192}, num_warps=4)
    compile_kernel(_first_best,
        {'best_value': '*fp32', 'best_index': '*i64', 'out': '*i64'},
        {'BLOCKS': blocks, 'BLOCK_B': 1 << (blocks-1).bit_length()}, num_warps=1)
print(f'E07 {cases} exact-index cases and six SM90 compilations passed')
