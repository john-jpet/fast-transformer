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

from kernels.pdl import wait as pdl_wait
from kernels.tune import register


@triton.jit
def _block_argmax(
    logits, best_value, best_index,
    VOCAB: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    pdl_wait()  # before any global memory access
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
    pdl_wait()  # before any global memory access
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


@triton.jit
def _fused_block_argmax(
    x_ptr, weight_ptr, best_value, best_index,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, EVEN_M: tl.constexpr,
):
    """One vocabulary tile's logits, rounded to BF16 like the projection, reduced to (max, first index) per row.

    Transposed tile product: weight[BLOCK_N, K'] @ x^T[K', BLOCK_M], FP32
    accumulation over the whole of K in ascending chunks (one split), rounded
    once to BF16 - the values the separate projection would have stored - so
    the first-maximum rule gives torch.argmax's answer on those logits.
    """
    pdl_wait()  # before any global memory access
    tile = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, BLOCK_M)
    columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    w_ptrs = weight_ptr + columns[:, None] * K + k[None, :]
    x_ptrs = x_ptr + k[:, None] + rows[None, :] * K
    acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for step in range(0, K // BLOCK_K):
        weight = tl.load(w_ptrs)
        if EVEN_M:
            x = tl.load(x_ptrs)
        else:
            x = tl.load(x_ptrs, rows[None, :] < M, other=0)
        acc = tl.dot(weight, x, acc)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    logits = acc.to(tl.bfloat16).to(tl.float32)
    value, index = tl.max(logits, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(best_value + rows * N // BLOCK_N + tile, value, rows < M)
    tl.store(best_index + rows * N // BLOCK_N + tile, tile * BLOCK_N + index, rows < M)


_FUSED = {}


def fused_argmax(x, weight, block_n=64, block_k=128):
    """argmax(x @ weight.T) for <= 32 BF16 rows without materializing the logits; None if not applicable."""
    rows, k = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1), x.shape[-1]
    n = weight.shape[0]
    if rows > 32 or n % block_n or k % block_k or x.dtype != torch.bfloat16 or not weight.is_contiguous():
        return None
    flat = x.reshape(rows, k).contiguous()
    block_m = 16 if rows <= 16 else 32
    tiles = n // block_n
    best_value = torch.empty((rows, tiles), dtype=torch.float32, device=x.device)
    best_index = torch.empty((rows, tiles), dtype=torch.int64, device=x.device)
    out = torch.empty(x.shape[:-1], dtype=torch.int64, device=x.device)
    _fused_block_argmax[(tiles,)](
        flat, weight, best_value, best_index, M=rows, N=n, K=k,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m, EVEN_M=rows == block_m, num_warps=4, num_stages=2,
    )
    _first_best[(rows,)](best_value, best_index, out, BLOCKS=tiles, BLOCK_B=triton.next_power_of_2(tiles), num_warps=4)
    return out


def greedy_tokens(x, weight, linear):
    """Greedy token per row of x @ weight.T: the fused kernel where the captured pass prefers it, else project + argmax.

    The fused path is offered to ``DecodeState.refine`` as a knob (default off)
    once it agrees with the projection path on the row count in use.
    """
    rows = x.shape[0] * (x.shape[1] if x.dim() == 3 else 1)
    key = (x.device, rows, weight.shape[0], weight.shape[1])
    if key not in _FUSED:
        _FUSED[key] = False
        if 4 < rows <= 32 and not torch.cuda.is_current_stream_capturing():
            try:
                generator = torch.Generator(device=x.device).manual_seed(4242)
                probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                fused = fused_argmax(probe, weight)
                agrees = fused is not None and torch.equal(fused, argmax(linear(probe, weight)))
            except Exception as error:  # noqa: BLE001
                print(f"fused argmax skipped: {error!r}", flush=True)
                agrees = False
            if agrees:
                register(
                    ("fused_argmax",) + key[1:], rows, weight.shape[0] * weight.shape[1] // 36 + 1, [False, True],
                    lambda: _FUSED[key], lambda option: _FUSED.__setitem__(key, option),
                )
    if _FUSED[key]:
        fused = fused_argmax(x, weight)
        if fused is not None:
            return fused
    return argmax(linear(x, weight))
