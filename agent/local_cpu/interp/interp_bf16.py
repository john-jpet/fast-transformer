"""Make TRITON_INTERPRET=1 numerically honest for bfloat16 (import BEFORE launching kernels).

Stock interpreter (3.1-3.5) stores bf16 as raw uint16 and
  * does +,*,/,compare, unary math, tl.sum and tl.dot on the raw BITS  -> garbage
  * casts fp32->bf16 by TRUNCATING the mantissa (GPU rounds to nearest even).
Here every bf16 operand is widened exactly to float64, the op runs there, and a bf16
result is rounded once with torch (RTNE), which is what the GPU's fptrunc does.
tl.dot: inputs widened exactly, accumulated in the accumulator's dtype (fp32 for our GEMM).
"""
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np, torch
import triton.language as tl
from triton.runtime import interpreter as I

STOCK = os.environ.get('FASTY_STOCK') == '1'   # control: leave the interpreter unpatched
B, TH = I.InterpreterBuilder, I.TensorHandle
BF = tl.bfloat16

def widen(data):            # uint16 bits -> float64 values (exact)
    a = np.ascontiguousarray(data)
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).double().numpy().reshape(a.shape)

def narrow(values):         # float -> bf16 bits, round-to-nearest-even
    a = np.ascontiguousarray(np.asarray(values))
    t = torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    return t.reshape(a.shape)

def is_bf(h):
    return getattr(h.dtype, "scalar", h.dtype) == BF

def val(h):
    return widen(h.data) if is_bf(h) else h.data

_binary = B.binary_op
def binary_op(self, lhs, rhs, op):
    if not (is_bf(lhs) or is_bf(rhs)):
        return _binary(self, lhs, rhs, op)
    out = op(val(lhs), val(rhs))
    return TH(out if out.dtype == np.bool_ else narrow(out), lhs.dtype.scalar)
if not STOCK: B.binary_op = binary_op

_unary = B.unary_op
def unary_op(self, arg, op):
    if not is_bf(arg):
        return _unary(self, arg, op)
    return TH(narrow(op(widen(arg.data))), arg.dtype.scalar)
if not STOCK: B.unary_op = unary_op

_cast = B.cast_impl
def cast_impl(self, src, dst_type):
    s, d = src.dtype.scalar, dst_type.scalar
    if s == BF and d == BF:
        return TH(src.data, d)
    if d == BF:
        return TH(narrow(val(src)), d)
    if s == BF:
        return TH(widen(src.data).astype(I._get_np_dtype(dst_type)), d)
    return _cast(self, src, dst_type)
if not STOCK:
    B.cast_impl = cast_impl
    B.create_fp_to_fp = lambda self, src, dst_type, rounding_mode: cast_impl(self, src, dst_type)
for name in () if STOCK else ("create_si_to_fp", "create_ui_to_fp", "create_fp_to_si", "create_fp_to_ui", "create_fp_ext", "create_fp_trunc"):
    setattr(B, name, lambda self, src, dst_type: cast_impl(self, src, dst_type))

def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
    acc = val(d)
    kind = acc.dtype if not is_bf(d) else np.float64
    out = np.matmul(val(a).astype(kind), val(b).astype(kind)) + acc
    return TH(narrow(out) if is_bf(d) else out.astype(d.data.dtype), d.dtype.scalar)
if not STOCK: B.create_dot = create_dot

_sum = I.ReduceOps.sum
def reduce_sum(self, input):
    if input.dtype.scalar != BF:
        return _sum(self, input)
    total = np.sum(widen(input.handle.data), axis=self.axis, keepdims=self.keep_dims)
    return self.to_tensor(narrow(total), input.dtype)
if not STOCK: I.ReduceOps.sum = reduce_sum

# CPU torch builds: the engine's launch-option picker asks the CUDA stream.
torch.cuda.is_current_stream_capturing = lambda: False

# numpy >= 2.5 refuses int() on the interpreter's shape-(1,) scalars, which breaks
# `for start in range(tensor, tensor, CONST)` loops (GEMM tiles, attention intervals).
_patch_tensor = I._patch_lang_tensor
def _patch_lang_tensor(tensor):
    _patch_tensor(tensor)
    tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])
I._patch_lang_tensor = _patch_lang_tensor
