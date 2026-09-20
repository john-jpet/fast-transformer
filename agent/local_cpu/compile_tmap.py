"""The persistent-launch "tmap" / "tmap3" kinds through the real kernels.linear._project, compiled for cuda:90.

Like compile_tma.py: a recorder stands in for ``_tmap_gemm`` / ``_persist_trans_gemm`` and a stub
for the descriptor. Checks per real shape x rows: 1-D grid of persistent_programs(tiles) <= 132
programs, a BF16 [M, N] output (never a Split), the fail-safe twin on the SAME grid and constants,
strict mode raising, and in the TTGIR: the TMA copy inside the INNERMOST loop (the K loop, nested in
the tile loop), num_stages - 1 prefetch copies hoisted out of the K loop, one ordinary (x) load.
Reports ``metadata.shared`` and compile seconds for the TMA kernel and its twin.
"""
import itertools
import time

import torch

from offline_compile import compile_kernel
from kernels import linear
from kernels.merged import Split


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def record(m, n, k, config, descriptor, strict=False):
    tma, twin, real = Recorder(), Recorder(), (linear._tmap_gemm, linear._persist_trans_gemm, linear._tma_descriptor)
    linear._tmap_gemm, linear._persist_trans_gemm = tma, twin
    stub = torch.zeros(linear.TMA_SIZE, dtype=torch.int8) if descriptor else None
    linear._tma_descriptor = lambda weight, block_n, block_k: stub
    try:
        result = linear._project(torch.zeros((m, k), dtype=torch.bfloat16), torch.zeros((n, k), dtype=torch.bfloat16),
                                 config, split_ok=True, strict=strict)
    finally:
        linear._tmap_gemm, linear._persist_trans_gemm, linear._tma_descriptor = real
    assert not isinstance(result, Split) and result.dtype == torch.bfloat16 and tuple(result.shape) == (m, n)
    return tma.calls, twin.calls


def loop_census(ttgir):
    """TMA copies and tt.loads by scf.for nesting depth: {depth: (copies, loads)}."""
    depth, census, stack = 0, {}, []
    for line in ttgir.splitlines():
        text = line.strip()
        indent = len(line) - len(line.lstrip())
        if stack and text.startswith("}") and indent == stack[-1]:
            stack.pop()
            depth -= 1
        entry = census.setdefault(depth, [0, 0])
        entry[0] += text.count("async_tma_copy_global_to_local")
        entry[1] += text.count("tt.load")
        if "scf.for" in text:
            depth += 1
            stack.append(indent)
    return {key: tuple(value) for key, value in census.items() if any(value)}


failures = 0
seconds, shared = {}, {}
for kind, m, (n, k) in itertools.product(
        ("tmap", "tmap3"), (16, 32, 5), ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560))):
    configs = [config for config in linear._candidates(m, n, k) if config[0] == kind]
    assert len(configs) == 1 and configs[0][3] == 1, configs
    config = configs[0]
    tma_calls, twin_calls = record(m, n, k, config, descriptor=False)
    assert not tma_calls and len(twin_calls) == 1, "fallback must be the ordinary-load twin"
    fallback = twin_calls[0]
    try:
        record(m, n, k, config, descriptor=False, strict=True)
        raise AssertionError("strict mode must raise without a descriptor")
    except RuntimeError:
        pass
    (grid, args, kwargs), = record(m, n, k, config, descriptor=True)[0]
    constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
    tiles = n // 64
    assert grid == (constants["PROGRAMS"],) and grid[0] <= 132 and constants["N_TILES"] == tiles
    assert -(-tiles // grid[0]) == -(-tiles // 132), "deeper tile loops than 132 programs would need"
    assert constants["STEPS"] * constants["BLOCK_K"] == k and tiles * constants["BLOCK_N"] == n
    assert fallback[0] == grid and all(fallback[2][key] == value for key, value in kwargs.items() if key != "num_stages")
    assert fallback[2]["EVEN_N"] and fallback[2]["EVEN_K"] and tuple(args[2].shape) == (m, n) and args[2].dtype == torch.bfloat16
    try:
        began = time.time()
        out = compile_kernel(linear._tmap_gemm, {"x_ptr": "*bf16", "desc_ptr": "*i8", "out_ptr": "*bf16"}, constants,
                             num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
        seconds.setdefault(kind, []).append(time.time() - began)
        shared.setdefault(kind, set()).add((constants["BLOCK_M"], out.metadata.shared))
        ttgir = out.asm["ttgir"]
        copies = ttgir.count("async_tma_copy_global_to_local")
        bulk = out.asm["ptx"].count("cp.async.bulk.tensor.2d.shared")
        stores = ttgir.count("async_tma_copy_local_to_global")
        census = loop_census(ttgir)
        print("compiled", kind, m, n, k, grid, {key: constants[key] for key in ("PROGRAMS", "N_TILES", "STEPS", "EVEN_M", "WIDE")},
              f"shared={out.metadata.shared} tma copies={copies} stores={stores} tt.load={ttgir.count('tt.load')} "
              f"dot_async={ttgir.count('dot_async')} ptx bulk-tensor={bulk}", flush=True)
        print("   (tma copies, tt.load) by loop depth =", census, flush=True)
        # Depth 2 = the K loop inside the tile loop: one pipelined copy + the x load. Depth 1 = the tile
        # loop body outside the K loop: the hoisted prefetch copies (num_stages - 1). Depth 0: nothing.
        stages = kwargs["num_stages"]
        # With JIT-accurate alignment attributes the x load is pipelined too, so
        # the K loop may show 0 ordinary loads instead of 1.
        if census not in ({1: (stages - 1, 0), 2: (1, 1)}, {1: (stages - 1, 0), 2: (1, 0)}):
            failures += 1
            print("UNEXPECTED LOAD STRUCTURE", kind, m, n, k)
        if not copies or not bulk or stores or "dot_async" not in ttgir:
            failures += 1
            print("NO TMA LOWERING / NO ASYNC DOT", kind, m, n, k)
    except Exception as error:  # noqa: BLE001
        failures += 1
        print("COMPILE FAILED", kind, m, n, k, config, repr(error)[:300])
    if kind == "tmap":
        try:
            began = time.time()
            out = compile_kernel(linear._persist_trans_gemm, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"},
                                 {key: value for key, value in fallback[2].items() if key not in ("num_warps", "num_stages")},
                                 num_warps=4, num_stages=2)
            seconds.setdefault("twin", []).append(time.time() - began)
            shared.setdefault("twin", set()).add((constants["BLOCK_M"], out.metadata.shared))
        except Exception as error:  # noqa: BLE001
            failures += 1
            print("TWIN COMPILE FAILED", m, n, k, repr(error)[:300])
for kind, times in seconds.items():
    print(f"{kind}: {len(times)} compiles, mean {sum(times) / max(1, len(times)):.1f}s, max {max(times, default=0):.1f}s, "
          f"shared (BLOCK_M, bytes) = {sorted(shared[kind])}")
print("failures", failures)
