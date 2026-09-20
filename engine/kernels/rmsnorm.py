"""Qwen3's RMSNorm in Triton, preserving the reference's BF16 cast boundaries.

Installed by ``decode.optimize_model`` for hidden and per-head norms.
Reduction order can differ; this does not promise bitwise-identical outputs.
"""

import torch
import triton
import triton.language as tl

from kernels.pdl import wait as pdl_wait

from kernels.merged import load_merged, source
from kernels.tune import pick

#: One row must fit in one block. Qwen3 4B norms 2560 columns (hidden) and 128
#: (per-head q/k norm), so both land well inside this.
MAX_BLOCK = 8192


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    pdl_wait()  # before any global memory access
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * row_stride + cols

    # The reference reduces in fp32 over the whole row. Masked lanes load as
    # zero, so they contribute nothing to the sum of squares.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)

    # Cast placement, and the whole reason this file exists. The reference ends:
    #
    #     return self.weight * hidden_states.to(input_dtype)
    #
    # so the normalised value is rounded to bfloat16 *before* the weight
    # multiply, not after. Keeping the product in fp32 and rounding once at the
    # end is the obvious version, is strictly more accurate, and is wrong: it
    # computes a different function, and on some prompt it moves a logit further
    # than the 2.0 tie margin allows. Reorder arithmetic freely; do not
    # reformulate it.
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, matching ``Qwen3RMSNorm.forward``.

    ``x`` is any shape whose last dimension matches ``weight``; ``weight`` is
    the module's learned gain, in the same dtype as ``x``.
    """
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = rows.shape
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {n_cols} columns does not")
    out = torch.empty_like(rows)
    _rms_norm_kernel[(n_rows,)](
        rows,
        weight,
        out,
        rows.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=4 if block <= 4096 else 8,
    )
    return out.reshape(shape)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr, residual_ptr, w_ptr, out_ptr, sum_ptr,
    WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
    COUNT=1, SPLITS: tl.constexpr = 1,
):
    pdl_wait()  # before any global memory access
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < WIDTH
    offset = row * WIDTH + cols
    x = load_merged(x_ptr, offset, valid, COUNT, SPLITS).to(tl.float32)
    residual = tl.load(residual_ptr + offset, valid, other=0).to(tl.float32)
    # The residual addition is a BF16 tensor operation in the native layer.
    summed = (x + residual).to(tl.bfloat16)
    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / WIDTH
    normalized = (summed_fp32 * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(w_ptr + cols, valid, other=0).to(tl.float32)
    tl.store(sum_ptr + offset, summed, valid)
    tl.store(out_ptr + offset, normalized * weight, valid)


def add_rms_norm(x, residual, weight, eps):
    """Return (normalized sum, BF16 sum), without mutating either input."""
    x_data, splits, count, shape = source(x)
    assert shape == tuple(residual.shape)
    assert residual.dtype == weight.dtype == torch.bfloat16
    assert x_data.dtype == (torch.bfloat16 if splits == 1 else torch.float32)
    width = shape[-1]
    block = triton.next_power_of_2(width)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {width} columns does not")
    x_rows = x_data.reshape(-1, width).contiguous() if splits == 1 else x_data
    residual_rows = residual.reshape(-1, width).contiguous()
    out = torch.empty_like(residual_rows)
    summed = torch.empty_like(residual_rows)
    rows = residual_rows.shape[0]

    def launch(warps):
        _add_rms_norm_kernel[(rows,)](
            x_rows, residual_rows, weight, out, summed,
            WIDTH=width, EPS=eps, BLOCK=block, COUNT=count, SPLITS=splits, num_warps=warps,
        )

    # Decode rows are launch-bound: measure the block width once per shape.
    # Verify blocks run 16-64 rows; every width is the same kernel writing the
    # same values, and the captured pass judges the choice.
    warps = pick(("add_rms_norm", rows, width), 4, (1, 2, 8), launch) if rows <= 64 else 4
    launch(warps)
    return out.reshape(shape), summed.reshape(shape)


@triton.jit
def _embed_rms_norm_kernel(ids_ptr, table_ptr, w_ptr, y_ptr, h_ptr, n_cols, eps, BLOCK: tl.constexpr):
    """Row = the embedding of ids[row], copied out as the residual stream and normalized like _rms_norm_kernel."""
    pdl_wait()  # before any global memory access
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    token = tl.load(ids_ptr + row).to(tl.int64)
    raw = tl.load(table_ptr + token * n_cols + cols, mask=mask, other=0.0)
    tl.store(h_ptr + row * n_cols + cols, raw, mask=mask)
    x = raw.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)


def embed_rms_norm(token_ids, table, weight, eps):
    """(normalized, hidden) for the embedding rows of ``token_ids``: one launch instead of a gather and a norm."""
    ids = token_ids.reshape(-1).contiguous()
    n_cols = table.shape[1]
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK or not table.is_contiguous() or ids.dtype != torch.int64:
        raise ValueError("embedding rows must fit one block and the table must be contiguous")
    hidden = torch.empty((ids.numel(), n_cols), dtype=table.dtype, device=table.device)
    out = torch.empty_like(hidden)

    def launch(warps):
        _embed_rms_norm_kernel[(ids.numel(),)](
            ids, table, weight, out, hidden, n_cols, eps, BLOCK=block, num_warps=warps,
        )

    launch(pick(("embed_rms_norm", ids.numel(), n_cols), 4 if block <= 4096 else 8, (2, 8), launch))
    shape = (*token_ids.shape, n_cols)
    return out.reshape(shape), hidden.reshape(shape)
