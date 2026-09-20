"""CPU-only workaround for Triton 3.1's NumPy-1 C-extension memory operations.

Read/write through NumPy views of the same ctypes memory, without the binary
extension whose ABI is incompatible with this machine's NumPy 2 installation.
Only enabled explicitly by local tests; never shipped with the engine.
"""
import ctypes
import numpy as np
from triton.runtime import interpreter as I


def view_addresses(pointers, mask, dtype):
    addresses = np.asarray(pointers, dtype=np.uint64)[mask]
    if not addresses.size:
        return None, None
    low, high = int(addresses.min()), int(addresses.max())
    width = np.dtype(dtype).itemsize
    buffer = (ctypes.c_ubyte * (high - low + width)).from_address(low)
    view = np.frombuffer(buffer, dtype=dtype)
    return view, ((addresses - low) // width).astype(np.int64)


def load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
    dtype = ptrs.get_element_ty()
    numpy_dtype = I._get_np_dtype(dtype)
    live = np.broadcast_to(mask.data, ptrs.data.shape).astype(bool)
    result = np.zeros(ptrs.data.shape, dtype=numpy_dtype)
    if other is not None:
        result[...] = other.data
    source, indices = view_addresses(ptrs.data, live, numpy_dtype)
    if source is not None:
        result[live] = source[indices]
    return I.TensorHandle(result, dtype)


def store(self, ptrs, value, mask, cache_modifier, eviction_policy):
    live = np.broadcast_to(mask.data, ptrs.data.shape).astype(bool)
    target, indices = view_addresses(ptrs.data, live, value.data.dtype)
    if target is not None:
        target[indices] = np.broadcast_to(value.data, ptrs.data.shape)[live]


I.InterpreterBuilder.create_masked_load = load
I.InterpreterBuilder.create_masked_store = store
