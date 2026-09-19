"""The "tma" and "tmah" projection kinds through the real kernels.linear._project, compiled for cuda:90.

A recorder stands in for ``_tma_gemm`` and a stub for the descriptor, so the grid
and constants are what ``_project`` would launch on the GPU. For each real shape
the TTGIR must hold the TMA copy op and the PTX the bulk-tensor instruction: that
is the evidence the descriptor load was lowered to the Hopper TMA unit rather
than to ordinary global loads. Also checks that without a descriptor the same
config launches ``_trans_gemm`` (the fail-safe), and that strict mode raises.
"tmah" (four TMA weight tiles per program sharing one x-tile load): its fail-safe
is ``_hoist_trans_gemm`` on the SAME grid and constants, and its TTGIR must hold
four TMA copies and one ordinary (x) load per K-loop iteration.
"""
import itertools
import time

import torch

from offline_compile import compile_kernel
from kernels import linear


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


KERNELS = {"tma": ("_tma_gemm", "_trans_gemm"), "tma3": ("_tma_gemm", "_trans_gemm"), "tmah": ("_tmah_gemm", "_hoist_trans_gemm")}


def record(m, n, k, config, descriptor, strict=False):
    names = KERNELS[config[0]]
    tma, trans, real = Recorder(), Recorder(), tuple(getattr(linear, name) for name in names) + (linear._tma_descriptor,)
    setattr(linear, names[0], tma), setattr(linear, names[1], trans)
    # The real one needs CUDA (on this CPU it retires the kind, which is its own fail-safe).
    stub = torch.zeros(linear.TMA_SIZE, dtype=torch.int8) if descriptor else None
    linear._tma_descriptor = lambda weight, block_n, block_k: stub
    try:
        linear._project(torch.zeros((m, k), dtype=torch.bfloat16), torch.zeros((n, k), dtype=torch.bfloat16),
                        config, split_ok=True, strict=strict)
    finally:
        setattr(linear, names[0], real[0]), setattr(linear, names[1], real[1])
        linear._tma_descriptor = real[2]
    return tma.calls, trans.calls


def loop_ops(ttgir):
    """(TMA copies, ordinary loads) inside scf.for bodies of the TTGIR."""
    depth, copies, loads = 0, 0, 0
    for line in ttgir.splitlines():
        if depth:
            copies += line.count("async_tma_copy_global_to_local")
            loads += line.count("tt.load")
        if "scf.for" in line:
            depth += 1
        elif depth and line.strip().startswith("}"):
            depth -= 1
    return copies, loads


failures = 0
seconds = {"tma": [], "tma3": [], "tmah": []}
for kind, m, (n, k) in itertools.product(
        ("tma", "tmah"), (16, 32, 64, 5), ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560))):
    kinds = [config[0] for config in linear._candidates(m, n, k)]
    assert "trans" not in kinds and kinds.index("tma") == 1, kinds  # "tma" replaces "trans"
    assert "exact" not in kinds and kinds[1:4] == ["tma", "tmap", "tmap3"] and len(kinds) <= 6, kinds
    configs = [config for config in linear._candidates(m, n, k) if config[0] == kind]
    if kind == "tmah" and n % 256:
        assert not configs, "tmah needs whole programs of four tiles"
        continue
    assert len(configs) <= 1, configs  # a kind is listed at most once; tma3 is currently not offered
    config = configs[0]
    tiles = 4 if kind == "tmah" else 1
    # Fail-safe: no descriptor (this is a CPU) -> the trans kernel; strict (validation) -> an exception.
    tma_calls, trans_calls = record(m, n, k, config, descriptor=False)
    assert not tma_calls and len(trans_calls) == 1, "fallback must be the ordinary-load twin"
    fallback = trans_calls[0]
    try:
        record(m, n, k, config, descriptor=False, strict=True)
        raise AssertionError("strict mode must raise without a descriptor")
    except RuntimeError:
        pass
    (grid, args, kwargs), = record(m, n, k, config, descriptor=True)[0]
    constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
    assert grid[0] * tiles * constants["BLOCK_N"] == n and constants["SPLITS"] * constants["CHUNK"] == k
    assert constants["CHUNK"] % constants["BLOCK_K"] == 0 and constants.get("TILES", 1) == tiles
    # The fail-safe writes the same partial through the same grid: same split count, same constants, mask-free.
    assert fallback[0] == grid and grid[1] == config[3] == constants["SPLITS"], (fallback[0], grid)
    # (the prefetch depth is a launch option of the TMA kernel only, not part of the layout)
    assert all(fallback[2][key] == value for key, value in kwargs.items() if key != "num_stages"), "fallback constants differ"
    assert fallback[2]["EVEN_N"] and fallback[2]["EVEN_K"] and fallback[1][2].shape == args[2].shape
    assert tuple(args[2].shape) == ((config[3], m, n) if config[3] > 1 else (m, n))
    try:
        ptr = "*bf16" if constants["SPLITS"] == 1 else "*fp32"
        began = time.time()
        out = compile_kernel(getattr(linear, KERNELS[kind][0]), {"x_ptr": "*bf16", "desc_ptr": "*i8", "out_ptr": ptr}, constants,
                             num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
        seconds[kind].append(time.time() - began)
        copies = out.asm["ttgir"].count("async_tma_copy_global_to_local")
        bulk = out.asm["ptx"].count("cp.async.bulk.tensor.2d.shared")
        stores = out.asm["ttgir"].count("async_tma_copy_local_to_global")
        weight_loads = out.asm["ttgir"].count("tt.load")
        print("compiled", m, n, k, config, grid, {key: constants[key] for key in ("SPLITS", "CHUNK", "EVEN_M", "WIDE")},
              f"ttgir tma copies={copies} tma stores={stores} tt.load={weight_loads} ptx bulk-tensor={bulk}", flush=True)
        in_loop = loop_ops(out.asm["ttgir"])
        print("   in-loop (tma copies, tt.load) =", in_loop, flush=True)
        # The compiler pipelines the copies: one set ahead of the loop (step 0) and one set per
        # iteration (the next step's tiles, behind the dots). The x tile is the only ordinary load.
        # (num_stages sets the depth: stages - 1 sets ahead of the loop, one inside it)
        if (copies, weight_loads) != (kwargs["num_stages"] * tiles, 1) or in_loop != (tiles, 1):
            failures += 1
            print("UNEXPECTED LOAD STRUCTURE", kind, m, n, k)
        if not copies or not bulk or stores:
            failures += 1
            print("NO TMA LOWERING", m, n, k)
    except Exception as error:  # noqa: BLE001
        failures += 1
        print("COMPILE FAILED", m, n, k, config, repr(error)[:300])
for kind, times in seconds.items():
    print(f"{kind}: {len(times)} compiles, mean {sum(times) / max(1, len(times)):.1f}s, max {max(times, default=0):.1f}s")
print("failures", failures)
