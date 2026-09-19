"""kernels/gemm.py through the real kernels.linear._project: compile for cuda:90 and emulate.

A recorder stands in for the two kernels, so the grid and constants checked here
are exactly what `_project` would launch. For each launch: (1) offline-compile
with those constants; (2) replay the kernel's pointer arithmetic in numpy (masks
present only where the EVEN_* flags keep them) and require in-bounds loads,
every output cell of every split written exactly once, and the FP64 product.
"""
import itertools

import numpy as np
import torch

from offline_compile import compile_kernel
from kernels import linear


class Recorder:
    def __init__(self, kernel):
        self.kernel, self.calls = kernel, []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, kwargs))


def emulate(kind, grid, c, x, w):
    m, n, k = c["M"], c["N"], c["K"]
    out = np.full((c["SPLITS"], m, n), np.nan)
    writes = np.zeros((c["SPLITS"], m, n), int)
    tiles = c.get("TILES", 1)
    for program, split, part in itertools.product(range(grid[0]), range(grid[1]), range(tiles)):
        rows = np.arange(c["BLOCK_M"])
        cols = (program * tiles + part) * c["BLOCK_N"] + np.arange(c["BLOCK_N"])
        row_ok = rows < m if not c["EVEN_M"] else np.ones_like(rows, bool)
        col_ok = cols < n if not c["EVEN_N"] else np.ones_like(cols, bool)
        assert rows[row_ok].max() < m and (not col_ok.any() or cols[col_ok].max() < n), "unmasked out-of-bounds lane"
        acc = np.zeros((c["BLOCK_M"], c["BLOCK_N"]))
        for step in range(c["CHUNK"] // c["BLOCK_K"]):
            ks = split * c["CHUNK"] + step * c["BLOCK_K"] + np.arange(c["BLOCK_K"])
            k_ok = ks < k if not c["EVEN_K"] else np.ones_like(ks, bool)
            assert not k_ok.any() or ks[k_ok].max() < k, "unmasked out-of-bounds k"
            xt = np.where(row_ok[:, None] & k_ok[None, :], x[np.minimum(rows, m - 1)][:, np.minimum(ks, k - 1)], 0.0)
            wt = np.where(col_ok[:, None] & k_ok[None, :], w[np.minimum(cols, n - 1)][:, np.minimum(ks, k - 1)], 0.0)
            acc += xt @ wt.T
        for i, j in itertools.product(rows[row_ok], range(c["BLOCK_N"])):
            if col_ok[j]:
                out[split, i, cols[j]] = acc[i, j]
                writes[split, i, cols[j]] += 1
    assert (writes == 1).all(), "a cell was not written exactly once"
    return out.sum(0)


def emulate_hoist_trans(grid, c, x, w):
    """``_hoist_trans_gemm`` literally: its flat pointer expressions on flat buffers.

    Every lane whose mask is compiled out (EVEN_*) or true must be in bounds;
    every output cell of every split is written exactly once; FP64 product.
    "tmah" without a descriptor (always, on a CPU) launches this kernel.
    """
    m, n, k, tiles = c["M"], c["N"], c["K"], c["TILES"]
    block_m, block_n, block_k = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
    xf, wf = x.reshape(-1), w.reshape(-1)
    out, writes = np.full(c["SPLITS"] * m * n, np.nan), np.zeros(c["SPLITS"] * m * n, int)

    def load(flat, offsets, ok):
        assert offsets[ok].size == 0 or (0 <= offsets[ok].min() and offsets[ok].max() < flat.size), "out-of-bounds load"
        return np.where(ok, flat[np.clip(offsets, 0, flat.size - 1)], 0.0)

    for program, split in itertools.product(range(grid[0]), range(grid[1])):
        rows = np.arange(block_m)
        columns = program * (tiles * block_n) + np.arange(block_n)
        ks = split * c["CHUNK"] + np.arange(block_k)
        row_ok = np.broadcast_to(rows[None, :] < m if not c["EVEN_M"] else True, (block_k, block_m))
        w_ptrs = columns[:, None] * k + ks[None, :]
        x_ptrs = ks[:, None] + rows[None, :] * k
        w_step = block_n * k
        oks = [np.broadcast_to((columns[:, None] + t * block_n) < n if not c["EVEN_N"] else True, (block_n, block_k))
               for t in range(tiles)]
        accs = [np.zeros((block_n, block_m)) for _ in range(tiles)]
        for step in range(c["CHUNK"] // block_k):
            k_ok = (ks + step * block_k) < k if not c["EVEN_K"] else np.ones(block_k, bool)
            xt = load(xf, x_ptrs, k_ok[:, None] & row_ok)  # ONE x tile per K chunk
            for t in range(tiles):
                accs[t] += load(wf, w_ptrs + t * w_step, oks[t] & k_ok[None, :]) @ xt
            w_ptrs, x_ptrs = w_ptrs + block_k, x_ptrs + block_k
        o_ptrs = split * (m * n) + columns[:, None] + rows[None, :] * n
        for t in range(tiles):
            ok = oks[t][:, :1] & row_ok[:1, :]
            offsets = (o_ptrs + t * block_n)[ok]
            assert offsets.size == 0 or (0 <= offsets.min() and offsets.max() < out.size), "out-of-bounds store"
            out[offsets] = accs[t][ok]
            np.add.at(writes, offsets, 1)
    assert (writes == 1).all(), "a cell was not written exactly once"
    return out.reshape(c["SPLITS"], m, n).sum(0)


def emulate_persist(grid, c, x, w):
    """``_persist_trans_gemm`` literally: a 1-D grid, per program the tile loop around the whole K loop.

    Flat pointer expressions on flat buffers; every unmasked lane in bounds; every cell of the
    single [M, N] output written exactly once ACROSS the programs (no splits); FP64 product.
    "tmap" / "tmap3" without a descriptor (always, on a CPU) launch this kernel.
    """
    m, n, k = c["M"], c["N"], c["K"]
    block_m, block_n, block_k = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
    assert len(grid) == 1 and grid[0] == c["PROGRAMS"] <= 132 and "SPLITS" not in c
    assert c["STEPS"] * block_k >= k > (c["STEPS"] - 1) * block_k and c["N_TILES"] * block_n >= n > (c["N_TILES"] - 1) * block_n
    xf, wf = x.reshape(-1), w.reshape(-1)
    out, writes = np.full(m * n, np.nan), np.zeros(m * n, int)

    def load(flat, offsets, ok):
        assert offsets[ok].size == 0 or (0 <= offsets[ok].min() and offsets[ok].max() < flat.size), "out-of-bounds load"
        return np.where(ok, flat[np.clip(offsets, 0, flat.size - 1)], 0.0)

    depths = []
    for pid in range(grid[0]):
        rows, lanes, ks = np.arange(block_m), np.arange(block_n), np.arange(block_k)
        row_ok = np.broadcast_to(rows[None, :] < m if not c["EVEN_M"] else True, (block_k, block_m))
        mine = range(pid, c["N_TILES"], c["PROGRAMS"])
        depths.append(len(mine))
        for tile in mine:
            columns = tile * block_n + lanes
            assert c["WIDE"] or columns.max() * k + k < 2 ** 31, "int32 offsets overflow"
            col_ok = np.broadcast_to(columns[:, None] < n if not c["EVEN_N"] else True, (block_n, block_k))
            w_ptrs = columns[:, None] * k + ks[None, :]
            x_ptrs = ks[:, None] + rows[None, :] * k
            acc = np.zeros((block_n, block_m))
            for step in range(c["STEPS"]):
                k_ok = (ks + step * block_k) < k if not c["EVEN_K"] else np.ones(block_k, bool)
                acc += load(wf, w_ptrs, col_ok & k_ok[None, :]) @ load(xf, x_ptrs, k_ok[:, None] & row_ok)
                w_ptrs, x_ptrs = w_ptrs + block_k, x_ptrs + block_k
            ok = col_ok[:, :1] & row_ok[:1, :]
            offsets = (columns[:, None] + rows[None, :] * n)[ok]
            assert offsets.size == 0 or (0 <= offsets.min() and offsets.max() < out.size), "out-of-bounds store"
            out[offsets] = acc[ok]
            np.add.at(writes, offsets, 1)
    assert (writes == 1).all(), "a cell was not written exactly once"
    assert max(depths) - min(depths) <= 1, "unbalanced residue classes"
    return out.reshape(m, n)


def launches(m, n, k, config):
    recorders = {name: Recorder(getattr(linear, name))
                 for name in ("_exact_gemm", "_trans_gemm", "_hoist_gemm", "_hoist_trans_gemm", "_tmah_gemm",
                             "_persist_trans_gemm", "_tmap_gemm")}
    for name, recorder in recorders.items():
        setattr(linear, name, recorder)
    # No descriptor, as on a GPU before the eager pass. (The real builder needs CUDA: on a CPU it
    # retires the TMA kinds, and every later _candidates list would silently lose "tmah".)
    descriptor, linear._tma_descriptor = linear._tma_descriptor, lambda weight, block_n, block_k: None
    try:
        x = torch.zeros((m, k), dtype=torch.bfloat16)
        w = torch.zeros((n, k), dtype=torch.bfloat16)
        linear._project(x, w, config, split_ok=True)
    finally:
        linear._tma_descriptor = descriptor
        for name, recorder in recorders.items():
            setattr(linear, name, recorder.kernel)
    (name, recorder), = [(a, r) for a, r in recorders.items() if r.calls]
    (grid, kwargs), = recorder.calls
    return recorder.kernel, grid, kwargs


failures = 0
# Real shapes: qkv, o, gate_up, down, lm_head at verify-block row counts (even and ragged M).
for m, (n, k) in itertools.product((5, 16, 32, 48, 64), ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560))):
    for config in linear._candidates(m, n, k):
        if config[0] == "tma":
            config = ("trans",) + config[1:]  # what "tma" launches without a descriptor; "trans" itself is no longer listed beside it
        if config[0] not in ("exact", "trans", "hoist", "tmah"):
            continue
        kernel, grid, kwargs = launches(m, n, k, config)
        # No descriptor on a CPU: "tmah" must launch its ordinary-load twin on the hoisted grid.
        assert (kernel.__name__ == "_hoist_trans_gemm") == (config[0] == "tmah"), kernel.__name__
        assert config[0] != "tmah" or grid == (n // 256, config[3])
        constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
        assert constants["SPLITS"] * constants["CHUNK"] >= k and (constants["SPLITS"] - 1) * constants["CHUNK"] < k
        try:
            ptr = "*bf16" if constants["SPLITS"] == 1 else "*fp32"
            compile_kernel(kernel, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": ptr}, constants,
                           num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
            print("compiled", m, n, k, config, grid, {key: constants[key] for key in ("SPLITS", "CHUNK", "EVEN_M", "EVEN_N", "EVEN_K", "WIDE")})
        except Exception as error:  # noqa: BLE001
            failures += 1
            print("COMPILE FAILED", m, n, k, config, repr(error)[:300])

# Small shapes through the same _project constants: even and ragged on every axis
# (the last three: whole programs of four tiles, the only layout "tmah" is offered for).
rng = np.random.default_rng(0)
replayed = 0
for m, n, k in ((16, 128, 256), (5, 128, 256), (16, 100, 256), (32, 192, 384), (7, 70, 300), (16, 64, 128), (48, 100, 256), (64, 192, 384),
                (16, 512, 384), (5, 256, 256), (32, 768, 640)):
    for kind in ("exact", "trans", "hoist", "tmah"):
        for splits in (1, 2, 3):
            config = (kind, 64, 128, splits, 4)
            if (splits - 1) * (-(-k // (splits * 128)) * 128) >= k:
                continue  # a dead split: exact_splits never produces one
            kernel, grid, kwargs = launches(m, n, k, config)
            x, w = rng.standard_normal((m, k)), rng.standard_normal((n, k))
            got = emulate(kind, grid, kwargs, x, w)
            if not np.allclose(got, x @ w.T):
                failures += 1
                print("EMULATION MISMATCH", m, n, k, config)
            if kind == "tmah":
                assert kernel.__name__ == "_hoist_trans_gemm" and kwargs["TILES"] == 4
                replayed += 1
                if not np.allclose(emulate_hoist_trans(grid, kwargs, x, w), x @ w.T, rtol=1e-12, atol=1e-12):
                    failures += 1
                    print("POINTER REPLAY MISMATCH", m, n, k, config)
# "tmap": the persistent 1-D launch. Small shapes (even and ragged on every axis) with the real 132-SM
# rule (one tile per program) and with 5 SMs (tile loops 1-3 deep, uneven residue classes); then two
# real shapes: gate_up (102 programs x 3 tiles) and lm_head (132 programs x 17-18 tiles), ragged M.
from kernels import gemm
persisted = 0
real_rule = linear.persistent_programs
for sms, shapes in ((132, ((16, 128, 256), (5, 128, 256), (16, 100, 256), (32, 192, 384), (7, 70, 300), (16, 64, 128), (32, 768, 640))),
                    (5, ((16, 512, 384), (5, 256, 256), (32, 768, 640), (7, 710, 300), (20, 1000, 517), (16, 64, 128))),
                    (132, ((16, 19456, 2560), (5, 151936, 2560)))):
    linear.persistent_programs = lambda tiles, sms=sms: gemm.persistent_programs(tiles, sms)
    try:
        for m, n, k in shapes:
            kernel, grid, kwargs = launches(m, n, k, ("tmap", 64, 128, 1, 4))
            assert kernel.__name__ == "_persist_trans_gemm" and grid == (gemm.persistent_programs(-(-n // 64), sms),)
            x, w = rng.standard_normal((m, k)), rng.standard_normal((n, k))
            persisted += 1
            if not np.allclose(emulate_persist(grid, kwargs, x, w), x @ w.T, rtol=1e-11, atol=1e-11):
                failures += 1
                print("PERSISTENT POINTER REPLAY MISMATCH", m, n, k, grid)
            else:
                print("persistent replay ok", (m, n, k), "grid", grid, "tiles", kwargs["N_TILES"], "steps", kwargs["STEPS"], flush=True)
    finally:
        linear.persistent_programs = real_rule
print("persistent pointer replays", persisted)
assert not linear._TMA_OFF[0] and replayed, "the TMA kinds were retired: tmah launches were not checked"
print("hoist-trans pointer replays", replayed)
print("failures", failures)
