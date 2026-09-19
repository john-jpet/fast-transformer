# CPU-only checks (no CUDA on this Mac)

Image: `fasty-cpucheck:3.1.0` (see `Dockerfile`; Torch 2.5.1 CPU, Triton 3.1.0).
Mount only `engine/` and this directory:

```sh
docker run --rm --platform linux/amd64 -v "$PWD/engine":/work/engine:ro \
  -v "$PWD/agent/local_cpu":/scratch:ro -e PYTHONPATH=/work/engine:/scratch \
  fasty-cpucheck:3.1.0 python /scratch/compile_packed.py
```

- `offline_compile.py`: `triton.compile` with an explicit `GPUTarget("cuda", 90, 32)`.
  Catches Triton type/semantic errors for the H100 target through `cubin`.
  Signature entries must be in argument order (the helper sorts them).
- `compile_packed.py`, `test_pack.py`: packed-weight kernels and codec.
- `check_qk_offsets.py`: flat-offset algebra of the fused Q/K kernel.
- `compile_attention_tma.py`: `_block_partials_tma` (verify attention with the whole-prefix K/V tiles read
  through Hopper TMA tensor maps) for T x capacity x BLOCK_N x SPLITS; asserts from the TTGIR that the K and V
  copies are pipelined (one pair ahead of the loop per stage, one pair per iteration) and prints compile
  seconds. `STAGES=3` compiles the deeper pipeline. Its ordinary-load twin is proven bit-identical to
  `_block_partials` by `interp/run_attention_twin.py` (the interpreter has no descriptor-load op).

## Executing the kernels on the CPU (`interp/`)

`interp/all.sh` runs every engine kernel in Triton's interpreter (about one
minute, exit status 1 on any failure): the speculation kernels against the
`check_tree.py` emulation, RMSNorm / SwiGLU / QK-RoPE / paired gate-up and the
Split consumers bit-exact against Torch BF16 references, the GEMV/GEMM kinds
within 0.5 BF16 ulp of the exact product, block and decode attention within
1 ulp of the FP64 reference. Image: `docker build -t fasty-tritoninterp:3.5.0
-f interp/Dockerfile.arm64 interp` (native arm64, Triton 3.5.0).
`interp_bf16.py` patches three interpreter bugs: BF16 arithmetic on raw uint16
bits, a truncating FP32->BF16 cast (the GPU rounds to nearest even), and
tensor-valued `range` bounds under new numpy. The earlier "returns zeros"
failure in the 3.1.0 image was numpy 2 breaking Triton 3.1's load/store
extension (numpy 1.26.4 fixes it), not the AMD64 emulation.

It proves index/mask/semantic correctness of what the kernel source says. It
says nothing about GPU compilation (use the offline compile scripts), races
between programs, FP32 reduction order on the GPU, or speed.

`compile_packed.py`, `count_loads.py` and `test_pack.py` exercise the archived
lossless 12-bit experiment (`agent/archive/packed_lossless_12bit.py`), which is
no longer in `engine/`; mount it as `kernels/packed.py` to rerun them.
`compile_words.py` targets the word-load kernels of commit `072d3e1`.
