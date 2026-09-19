from offline_compile import compile_kernel
from kernels.pdl import _produce, _consume
a = compile_kernel(_produce, {"out_ptr": "*i32", "value_ptr": "*i32"}, {"COUNT": 1024}, num_warps=4)
b = compile_kernel(_consume, {"in_ptr": "*i32", "out_ptr": "*i32"}, {"COUNT": 1024}, num_warps=4)
assert "griddepcontrol.wait" in a.asm["ptx"] and "griddepcontrol.wait" in b.asm["ptx"]
print("self-test kernels compile for cuda:90")
