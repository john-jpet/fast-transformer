from offline_compile import compile_kernel
import decode
out = compile_kernel(decode._nothing, {}, {"SEQ": 0}, num_warps=1, num_stages=2)
ptx = out.asm["ptx"]
print(f"noop kernel: shared={out.metadata.shared} ptx_lines={len(ptx.splitlines())} "
      f"ld.global={ptx.count('ld.global')} st.global={ptx.count('st.global')}")
