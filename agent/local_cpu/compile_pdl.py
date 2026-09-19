"""PDL: every engine kernel's PTX must start its work with griddepcontrol.wait; the launcher patch must apply."""
import os
from offline_compile import compile_kernel
from kernels.rmsnorm import _rms_norm_kernel
from kernels.argmax import _first_best
from kernels.spec import _relocate
from kernels import pdl
out = compile_kernel(_rms_norm_kernel, {"x_ptr": "*bf16", "w_ptr": "*bf16", "y_ptr": "*bf16", "row_stride": "i32", "n_cols": "i32", "eps": "fp32"}, {"BLOCK": 4096}, num_warps=4)
assert "griddepcontrol.wait" in out.asm["ptx"], "no wait in rms_norm PTX"
out = compile_kernel(_first_best, {"best_value": "*fp32", "best_index": "*i64", "out": "*i64"}, {"BLOCKS": 19, "BLOCK_B": 32}, num_warps=1)
assert "griddepcontrol.wait" in out.asm["ptx"]
out = compile_kernel(_relocate, {"store": "*bf16", "move_from": "*i64", "move_to": "*i64"}, {"BATCH": 2, "KV_HEADS": 8, "CAPACITY": 553, "DIM": 128, "BLOCK_H": 8}, num_warps=1)
assert "griddepcontrol.wait" in out.asm["ptx"]
ptx = out.asm["ptx"]
first_mem = min(i for i in (ptx.find("ld.global"), ptx.find("st.global")) if i >= 0)
assert ptx.find("griddepcontrol.wait") < first_mem, "the wait must precede the first global access"
# launcher patch: generate the stock and patched C source for one signature
from triton.backends.nvidia import driver as nv
stock = nv.make_launcher({}, {0: "*bf16", 1: "i32"}, {"ids_of_const_exprs": ()})
assert pdl.install()
patched = nv.make_launcher({}, {0: "*bf16", 1: "i32"}, {"ids_of_const_exprs": ()})
assert "CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION" in patched and patched != stock and "getenv(\"FASTY_PDL\")" in patched
assert patched.count("cuLaunchKernelEx_t") >= 2 and "#include <stdlib.h>" in patched
print("PDL: waits precede global accesses; launcher patch applies")
