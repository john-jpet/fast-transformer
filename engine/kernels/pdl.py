"""Programmatic Dependent Launch for the engine's Triton kernels (Hopper, CUDA 12.3+).

A verify pass is ~350 small kernels replayed from one CUDA graph. Between two
dependent kernels the GPU drains the first completely, then launches the
second: 1-2 us of idle ramp per boundary, ~0.3-0.5 ms of a 4.3 ms pass. With
PDL the next kernel may launch while its predecessor finishes; it blocks at
``griddepcontrol.wait`` until every prior kernel has completed and flushed, so
placed before the first global load or store the arithmetic and its results are
unchanged. ``griddepcontrol.wait`` is a no-op in a kernel launched without the
attribute, and kernels without the wait (PyTorch's own, cuBLAS) are launched
without it and keep full dependencies.

Two halves: ``wait()`` is the first statement of every engine kernel; the
launcher patch below makes Triton 3.1's generated C launch every single-CTA
kernel with CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION while the
environment flag is set. Stream capture turns that attribute into a
programmatic graph edge. ``self_test`` decides at warmup, on a two-kernel
graph, whether to keep the flag; anything unexpected clears it.
"""

import os

import torch
import triton
import triton.language as tl

FLAG = "FASTY_PDL"
#: Measured on the platform (candidate 80): the batch-1 pass got 13% SLOWER
#: and batch 4 5% slower with every kernel launched programmatically - the
#: early-launched blocks occupy SMs while spinning at their wait and starve the
#: bandwidth-bound kernel still running (llama.cpp saw the same for matvec).
#: The wait instruction stays in the kernels (a no-op without the attribute).
ENABLED = False
#: The interpreter cannot execute inline PTX: compile the wait away there.
ASM: tl.constexpr = os.environ.get("TRITON_INTERPRET") != "1"


@triton.jit
def wait():
    """Block until every earlier kernel of the stream has completed and flushed."""
    if ASM:
        tl.inline_asm_elementwise("griddepcontrol.wait; // $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


_STOCK = None
_ATTRIBUTE = '''    if (num_ctas == 1) {{
      static int pdl = -1;
      if (pdl < 0) {{ pdl = getenv("FASTY_PDL") != NULL; }}
      if (!pdl) {{
        CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
      }} else {{
        CUlaunchAttribute pdlAttr[1];
        pdlAttr[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        pdlAttr[0].value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig pdlConfig;
        pdlConfig.gridDimX = gridX;
        pdlConfig.gridDimY = gridY;
        pdlConfig.gridDimZ = gridZ;
        pdlConfig.blockDimX = 32 * num_warps;
        pdlConfig.blockDimY = 1;
        pdlConfig.blockDimZ = 1;
        pdlConfig.sharedMemBytes = shared_memory;
        pdlConfig.hStream = stream;
        pdlConfig.attrs = pdlAttr;
        pdlConfig.numAttrs = 1;
        static cuLaunchKernelEx_t pdlLaunch = NULL;
        if (pdlLaunch == NULL) {{
          pdlLaunch = getLaunchKernelExHandle();
        }}
        CUDA_CHECK(pdlLaunch(&pdlConfig, function, params, 0));
      }}
    }} else {{'''
_STOCK_BRANCH = '''    if (num_ctas == 1) {{
      CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
    }} else {{'''


def install():
    """Patch Triton's launcher generator once, before any kernel is launched. Returns whether it applied."""
    global _STOCK
    if _STOCK is not None:
        return True
    try:
        from triton.backends.nvidia import driver

        stock = driver.make_launcher

        def make_launcher(constants, signature, ids):
            source = stock(constants, signature, ids)
            if _STOCK_BRANCH.replace("{{", "{").replace("}}", "}") not in source:
                return source
            patched = source.replace(
                _STOCK_BRANCH.replace("{{", "{").replace("}}", "}"),
                _ATTRIBUTE.replace("{{", "{").replace("}}", "}"),
            )
            return patched.replace("#include <dlfcn.h>", "#include <dlfcn.h>\n#include <stdlib.h>", 1)

        driver.make_launcher = make_launcher
        _STOCK = stock
        return True
    except Exception as error:  # noqa: BLE001 - PDL is optional
        print(f"PDL launcher patch unavailable: {error!r}", flush=True)
        return False


@triton.jit
def _produce(out_ptr, value_ptr, COUNT: tl.constexpr):
    wait()
    offsets = tl.arange(0, COUNT)
    tl.store(out_ptr + offsets, tl.load(value_ptr) + offsets)


@triton.jit
def _consume(in_ptr, out_ptr, COUNT: tl.constexpr):
    wait()
    offsets = tl.arange(0, COUNT)
    tl.store(out_ptr + offsets, tl.load(in_ptr + offsets) * 2)


def self_test(device, count=1024):
    """Keep PDL only if a captured two-kernel chain replays bit-exactly with it.

    Runs before any engine kernel exists. A driver that cannot capture the
    attribute raises from the launcher (a RuntimeError, never an abort);
    a misordered result would show as the wrong sum. Either clears the flag.
    """
    if not ENABLED or not torch.cuda.is_available() or not install():
        return False
    os.environ[FLAG] = "1"
    try:
        stage = torch.zeros(count, dtype=torch.int32, device=device)
        out = torch.zeros(count, dtype=torch.int32, device=device)
        value = torch.zeros((), dtype=torch.int32, device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _produce[(1,)](stage, value, COUNT=count, num_warps=4)
                _consume[(1,)](stage, out, COUNT=count, num_warps=4)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(8):
                _produce[(1,)](stage, value, COUNT=count, num_warps=4)
                _consume[(1,)](stage, out, COUNT=count, num_warps=4)
        expected = torch.arange(count, dtype=torch.int32, device=device)
        for trial in range(20):
            value.fill_(trial)
            graph.replay()
            torch.cuda.synchronize(device)
            if not torch.equal(out, (expected + trial) * 2):
                raise RuntimeError("programmatic launch reordered a dependent kernel")
        print("PDL: programmatic dependent launch is on", flush=True)
        return True
    except Exception as error:  # noqa: BLE001
        os.environ.pop(FLAG, None)
        uninstall()
        print(f"PDL: off ({error!r})", flush=True)
        return False


def uninstall():
    """Back to Triton's own launcher generator for every kernel compiled from now on."""
    global _STOCK
    if _STOCK is not None:
        from triton.backends.nvidia import driver

        driver.make_launcher = _STOCK
        _STOCK = None
