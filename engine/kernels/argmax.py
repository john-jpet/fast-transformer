"""Greedy token ids straight from BF16 logits: torch.argmax's answer in two small launches.

Stage one reduces each (row, vocabulary block) to its maximum and that
maximum's first index; stage two picks each row's first best block. Ties
resolve to the lowest index at both stages, which is torch.argmax's rule, and
BF16 -> FP32 is exact, so the result is identical. torch's generic indexed
reduction over 151936 columns was the largest remaining PyTorch kernel in a
verify pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_argmax(
    logits, best_value, best_index,
    VOCAB: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * VOCAB + columns, columns < VOCAB, other=float("-inf")).to(tl.float32)
    value, index = tl.max(values, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(best_value + row * BLOCKS + block, value)
    tl.store(best_index + row * BLOCKS + block, block * BLOCK + index)


@triton.jit
def _first_best(
    best_value, best_index, out,
    BLOCKS: tl.constexpr, BLOCK_B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.arange(0, BLOCK_B)
    values = tl.load(best_value + row * BLOCKS + blocks, blocks < BLOCKS, other=float("-inf"))
    _, chosen = tl.max(values, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(out + row, tl.load(best_index + row * BLOCKS + chosen))


def argmax(logits, block=8192):
    """int64 argmax over the last dimension of BF16 logits [..., V]."""
    if logits.dtype != torch.bfloat16:
        return logits.argmax(dim=-1)
    logits = logits.contiguous()
    vocab = logits.shape[-1]
    rows = logits.numel() // vocab
    blocks = triton.cdiv(vocab, block)
    best_value = torch.empty((rows, blocks), dtype=torch.float32, device=logits.device)
    best_index = torch.empty((rows, blocks), dtype=torch.int64, device=logits.device)
    out = torch.empty(logits.shape[:-1], dtype=torch.int64, device=logits.device)
    _block_argmax[(rows, blocks)](logits, best_value, best_index, VOCAB=vocab, BLOCKS=blocks, BLOCK=block, num_warps=4)
    _first_best[(rows,)](best_value, best_index, out, BLOCKS=blocks, BLOCK_B=triton.next_power_of_2(blocks), num_warps=1)
    return out
