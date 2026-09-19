"""``_block_partials_tma`` for cuda:90: the K and V prefix tiles must lower to pipelined Hopper TMA copies.

For each (T, capacity, BLOCK_N, SPLITS) compiles the TMA kernel and its ordinary-load twin and reports, from
the TTGIR, the TMA copies in total and inside the ``scf.for`` bodies (a pipelined loop holds one K+V pair ahead
of the loop and one pair per iteration) and, from the PTX, the bulk-tensor copy instruction. Arguments: optional
``T,capacity,block_n,splits`` to compile one variant only (a compiler segfault cannot be caught in-process).
"""
import itertools
import os
import sys
import time

from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials_tma

PTRS = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "kd_ptr": "*i8", "vd_ptr": "*i8", "position_ptr": "*i64",
        "chain_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}


STAGES = int(os.environ.get("STAGES", "2"))  # the launcher uses 2


def loop_ops(ttgir):
    """(TMA copies, ordinary loads) inside scf.for bodies of the TTGIR."""
    depth, copies, loads = 0, 0, 0
    for line in ttgir.splitlines():
        if depth:
            copies += line.count("async_tma_copy_global_to_local")
            loads += line.count("tt.load")
        if "scf.for" in line:
            depth += 1
        elif depth and line.strip().startswith("}") and not line.strip().startswith("})"):  # "})" closes a tt.reduce body
            depth -= 1
    return copies, loads


def build(tokens, capacity, block_n, splits, tma, batch=16):
    sig = dict(PTRS)
    if splits == 1:
        sig["partial_ptr"] = sig["stats_ptr"] = "*bf16"
    if not tma:
        sig["kd_ptr"] = sig["vd_ptr"] = "*bf16"
    rows = batch * 8 * capacity
    constants = {
        "TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": capacity,
        "SPLITS": splits, "CHUNK": -(-capacity // splits), "SCALE": 128 ** -0.5,
        "BLOCK_M": max(16, 1 << (tokens * 4 - 1).bit_length()), "BLOCK_N": block_n,
        "LIMIT": capacity - rows % block_n, "TMA": tma,
    }
    began = time.time()
    out = compile_kernel(_block_partials_tma, sig, constants, num_warps=4, num_stages=STAGES)
    return out, time.time() - began


if len(sys.argv) > 1:
    variants = [tuple(int(part) for part in sys.argv[1].split(","))]
else:
    variants = list(itertools.product((2, 4, 8, 16), (549, 2086, 640, 704), (64, 128), (1, 4)))
failures, tma_seconds, twin_seconds = 0, [], []
for tokens, capacity, block_n, splits in variants:
    out, seconds = build(tokens, capacity, block_n, splits, True)
    tma_seconds.append(seconds)
    ttgir, ptx = out.asm["ttgir"], out.asm["ptx"]
    copies, bulk = ttgir.count("async_tma_copy_global_to_local"), ptx.count("cp.async.bulk.tensor.2d.shared")
    stores = ttgir.count("async_tma_copy_local_to_global")
    in_loop = loop_ops(ttgir)
    waits = ttgir.count("wait_barrier")
    print(f"T={tokens} C={capacity} BLOCK_N={block_n} SPLITS={splits}: {seconds:.1f}s tma copies={copies} "
          f"in-loop(copies, tt.load)={in_loop} wait_barrier={waits} tma stores={stores} ptx bulk-tensor={bulk}", flush=True)
    if len(sys.argv) > 2:
        print(ttgir)
    # Pipelined: K and V copies ahead of the loop (first tile) plus K and V inside it (next tile).
    if copies < 4 or in_loop[0] < 2 or not bulk or stores:
        failures += 1
        print("   NOT PIPELINED TMA", flush=True)
    _, seconds = build(tokens, capacity, block_n, splits, False)
    twin_seconds.append(seconds)
print(f"tma: {len(tma_seconds)} compiles, mean {sum(tma_seconds) / len(tma_seconds):.1f}s, max {max(tma_seconds):.1f}s; "
      f"twin: mean {sum(twin_seconds) / len(twin_seconds):.1f}s, max {max(twin_seconds):.1f}s")
print("failures", failures)
