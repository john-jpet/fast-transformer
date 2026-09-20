"""Compile engine Triton kernels for an H100 target without a GPU (type/semantic check only)."""
import sys, time
sys.path.insert(0, "/work/engine")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import AttrsDescriptor

TARGET = GPUTarget("cuda", 90, 32)

def compile_kernel(fn, signature, constants, divisible=(), **options):
    """Compile as the JIT would: every pointer, and every integer named in
    ``divisible``, carries tt.divisibility=16.

    The launcher derives this from the real addresses and values, and the
    compiler's vectoriser and its software pipeliner both refuse to act
    without it (`vec = min(ptrContiguity, maskAlignment)`; `vec * bitwidth <
    32` is dropped). A compile without it reports scalar loads for code that
    vectorises perfectly well on the device, so this is not optional.
    """
    names = fn.arg_names
    ordered = sorted(signature.items(), key=lambda item: names.index(item[0]))
    indices = {names.index(k): v for k, v in ordered}
    aligned = {index for index, ty in indices.items() if str(ty).startswith("*")}
    aligned |= {names.index(k) for k in divisible}
    src = ASTSource(fn=fn, signature=indices,
                    constants={names.index(k): v for k, v in constants.items()},
                    attrs=AttrsDescriptor(divisible_by_16=aligned))
    t = time.time()
    out = triton.compile(src, target=TARGET, options=options)
    print(f"compiled {fn.__name__}: stages={list(out.asm.keys())} in {time.time()-t:.1f}s", flush=True)
    return out

if __name__ == "__main__":
    from kernels.swiglu import _swiglu
    compile_kernel(_swiglu, {"packed": "*bf16", "output": "*bf16"}, {"WIDTH": 9728, "BLOCK": 1024}, num_warps=4)
