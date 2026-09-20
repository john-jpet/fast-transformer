# Plan of attack — 2026-09-19 17:05 UTC

State: #1 at 1095.1 (c48 `b1ca1cc`), dryfter 1087.3, third 1027.1 and rising.
One official run = 12-15 min, FIFO, whole run capped at 900 s (c48: 710 s;
c50: 799 s; c52 canceled at 917 s). Identical code varies +/-1%.
Sources: `~/.cache/fasty-lab/plan/*.md` (12 subagent reports: cost model, pacing,
lab experiment on the real model, 8 sourced web reports).

## What the research says (one paragraph each)

**Pass cost is near its ceiling.** Our verify pass is ~4.3 ms at 16 rows = 1.87
TB/s effective. 8B models reach 2.2-2.76 TB/s, but fused 2.5K-wide stacks only
1.6-1.75: for Qwen3-4B the cuBLAS/Triton floor is 4.1-4.5 ms. Remaining exact
kernel work is worth 4-7% of a pass in total (argmax 1-2% at 16 rows, 3-5% at
64; folding QK-norm+RoPE+KV-write into the block attention kernel 2.5-4%; small
fusions <0.5% each). Attention already uses BF16 dots with FP32 statistics.
Prefill is at parity with the best stacks at 4x2048 (55% MFU); 1x512 is only
37% MFU (10.5 ms vs 7.5-8.5 plausible). cuDNN SDPA could cut 4x2048 TTFT ~5%.

**Pacing is fine.** It forfeits 0% at batch 4/16 and 1.3-3.3% at batch 1 on
platform-like text; no adaptive rule beats the static floor; do not lower it
(c29's public-0 samples spread 24.6%). Watch item: floor 0.60 for outputs >= 96
at batch <= 2 (simulated 11-27% gate risk on hard text; zero platform failures
in ~15 runs). Raise to 0.64 on the first `unstable_timing`.

**The lever is tokens per pass.** We are PLD-class (1.33-1.7 tokens/pass);
model-free ceiling in the literature is Token Recycling / SAM+TR at 2.2-3.0
tokens/pass — but those rely on a table warmed ACROSS requests, which our rules
forbid; within one 32-128 token generation our offline test found recycling
dead. Literature confirms our tree shape for matched rows (chain + depth-1
siblings; depth-2 breadth buys nothing there) but challenges it for table-only
rows (bushy depth-2 trees). Cross-row budget allocation (TETRIS) maximises total
accepted tokens, whereas our generation ends with its SLOWEST row (lag-based
dealing already measured dead): low value. Layer-skip / early-exit / Jacobi /
KV-window self-drafting all lose to free n-gram drafts at our shapes.

## Ranked work list

In flight (each already passed: offline cuda:90 compile, Triton interpreter
execution, whole-engine CPU smoke test, unit tests):

| # | candidate | expected | risk |
|---|---|---|---|
| c53 `6334ff5` (measuring) | mask-free/transposed GEMM tiles, second block size inherits layouts, one-token shapes stay on cuBLAS | 0-2%, shorter warmup than c52 | run time (est. ~790 s) |
| c54 `6a30231` (queued) | + two-stage Triton argmax, tuning budgets 24->18 s, refine 14->10 s | +1-2% B=1, +3-5% at 64 rows; ~-40 s run time | low |
| c55 `13a81d0` (held) | previous pass's prediction after its first wrong draft = first sibling | lab: 2.0-2.9% fewer passes at T=16, 1.5-1.9% at T=4 | low (drafts only) |
| c56 `2f67fc3` (held) | speculation for batches 17-64, kept only if it beats the measured plain step | 0-8% on such shapes, 0 elsewhere | +8-15 s warmup on such shapes |

Next, in order (lab first where a lab test exists; one change per run unless
independent):

1. **Source-conditional tree for no-match rows** (lab, 1 h): when the suffix
   match is <= 1 token, spend the 15 slots as chain-of-table-walk + depth-1
   siblings + ONE continuation token on the best 2-3 siblings (nodes (1,2),
   (2,1): the literature's +8-11% tokens/pass over chain+leaves for weak
   drafters). Needs: `_propose` emits a parent index per slot instead of the
   chain/sibling dichotomy; `_block_partials` tree mask from parent pointers
   (depth <= 2 keeps it a two-term mask); `_settle` follows sibling->child;
   `_relocate` moves up to 2 slots. Gate: >= 4% fewer passes offline at B=1.
2. **LogitSpec-style re-rank of siblings by the bonus row's logits** (lab, 30
   min; `samples_512_128_ctxtop8.json` exists): gather logit[candidate] for the
   <= 16 candidates at the last accepted slot, order siblings by it. One gather,
   no wide kernel. Gate: >= 2% fewer passes.
3. **Fold QK-norm + RoPE + KV write into `_block_partials`** (2.5-4% of a pass,
   36 fewer launches): new kernel, medium risk; interpreter + smoke test cover it.
4. **cuDNN SDPA for prefill with a warmup self-check against FLASH and a
   permanent fallback** (+2% on 4x2048-like shapes, ~0 elsewhere): one run to
   learn whether 2.5.1 accepts native 8-head K/V; expanding KV kills the gain.
5. **Hidden-state similarity tie-break among history matches (PLD+)**: lab
   pre-test by dumping one mid-layer's states on MPS; GPU cost = one bmm + a
   52 MB buffer that relocates like KV. Gate: >= 3% fewer passes.
6. **1x512 prefill clean-up**: needs GPU timing to find the 2-3 ms; only with
   Modal/Lightning access or as blind single-change runs.

Rejected with reasons: trigram table at load (17-32 s x 6 workloads against the
900 s cap), TETRIS allocation (wrong objective for slowest-row-bound
generations), layer-skip/early-exit/Jacobi/KV-window drafting (arithmetic
loses to free drafts), lowering pace floors, more GEMM tile search, cuBLASLt,
paired gate/up block kernel, learned ranker kernel.

## Process rules (from today's failures)

- Every warmup second costs six (one per workload). Record run duration next to
  the score; stay under ~800 s.
- Before every push: `python3 -m unittest discover -s tests`, `dryft validate`,
  `agent/local_cpu/interp/all.sh`, `agent/local_cpu/smoke_engine.py`, the
  offline compile script for any new kernel/constexpr combination.
- Keep exactly one run measuring and at most one queued; hold the rest locally.
- A change under ~1.5% needs a second run or a public-case signal before `keep`.

## Updates after review (17:20 UTC)

- **Item 1 (depth-2 tree for no-match rows) is DEAD** - lab, 3 regimes, paired
  bootstrap (`~/.cache/fasty-lab/plan/depth2_tree.md`): best shape saves
  1.2-1.4% of passes at T=16 (bar: 4%) and every fixed child shape LOSES 2.7%
  at T=4. A child is right only 17% (table) / 35% (history) of the time when
  its sibling hits; ~40% would be needed. Do not build the parent-pointer kernel.
- **Codex review**: at batch 1 the median sample sits on the pacing floor
  (0.70 x pass time), so pass-TIME cuts convert ~1:1 while fewer passes barely
  show; acceptance work pays at batch >= 4. Item 3 (fused QK-norm/RoPE/KV-write
  block kernel) is now first; a subagent is building it in a worktree.
  Item 2 only makes sense where candidates exceed lanes (T=4/8) or when it
  changes draft 1: the running lab test measures exactly that increment over c55.
- c56 (batches 17-64) deferred: expected gain ~0 with conservative acceptance.
- c53 = 1097.7 (new best, 817 s). c54 (argmax + trimmed budgets) measuring;
  c57 (stale sibling + cuDNN prefill + frozen GC) queued.
- Optional blind single-change runs when the queue is otherwise empty:
  `CUBLAS_WORKSPACE_CONFIG`/`CUBLASLT_WORKSPACE_SIZE` = 32 MiB (NVIDIA's Hopper
  recommendation; affects prefill GEMMs and 64-row verify blocks; reorder-class).
- **Item 2 (logit re-ranking of siblings) is DEAD** - lab, 48 samples x 2
  regimes (`~/.cache/fasty-lab/plan/logit_rerank.md`): sorting siblings by the
  gathered logit is -0.1% at T=16; choosing draft 1 by it is HARMFUL (+1.8 to
  +2.8% passes); the increment over the shipped stale-guess sibling straddles 0
  at every block size. The successor table's top-8 already covers 38-40% of
  next-next tokens vs 22% for the logits' top-8. The same run re-confirmed the
  stale rule alone at -1.3 to -2.2% passes (48 samples).
- **Item 5 (hidden-state similarity for the copy source, PLD+) is DEAD** - lab,
  120 samples (`~/.cache/fasty-lab/plan/hidden_match.md`): the implementable
  variant saves 0.3-0.7% of passes at T=16 and 0.25% at T=4 (bar 3% / 2%). Even
  a PERFECT source selector saves only 2.8-5.1%: in 70-75% of the passes with a
  match of <= 1 token, NO earlier occurrence is followed by the right token.
  Conclusion: history-copy drafting is at its ceiling on this text; what is
  left on the acceptance side is a better context-aware predictor for no-match
  passes, which without trained weights we do not have. Remaining work is
  pass time (fusion kernel, attention prefix loop) and prefill (cuDNN).
- **Copy-logit siblings (RACER; the one new idea in the user's pasted research)
  are DEAD** - lab, real model, 84 samples (`~/.cache/fasty-lab/plan/copy_logit.md`):
  with logits available only at generated positions the best rule saves 1.2% of
  passes at T=16 and 0.2% at T=4 (bars 4% / 2%); even with prompt logits
  (+10% TTFT) it is 1.8% / 0.9%. Ranks 2-8 at the earlier occurrence hold the
  next token 25% of the time, but only 9-10% are tokens our follower, history
  and table siblings do not already offer. Draft-side ideas tested and dead
  today: depth-2 trees, logit re-ranking, hidden-state match selection, pair
  table, copy-logit. The draft side is exhausted for training-free methods on
  this text; remaining work is pass time, tuning coverage and warmup time.
- **Pipelining audit (19:20 UTC, offline cuda:90 compiles):** the PTX of
  `_exact_gemm`, `_skinny_gemm` and `_block_partials` is byte-identical for
  num_stages 1, 2, 3 and 4: ordinary pointer loads in our K/KV loops get NO
  asynchronous prefetch in Triton 3.1.0. The TMA kernels do (the tmah build saw
  4 descriptor copies hoisted ahead of the loop). So on this stack async
  prefetch = TMA descriptor loads inside an innermost loop. Static buffers that
  can have host-built descriptors: weights (done: tma/tmah) and the KV cache
  (`cache.store`, allocated once) -> TMA for attention K/V tiles is the next
  kernel (published: 1.07-1.17x on attention at batch 16, ~0 at batch 1).
  Activations live in graph-private pools whose addresses are only known at
  capture: no descriptors for x tiles.

## Correction to the deep-research plan (2026-09-19 22:30 UTC)

W2's premise is refuted, but in our favour. The offline harness never passed
`divisible_by_16`, so it measured a compiler that had been told the pointers
might be unaligned. With the attributes the JIT really passes:
- the tile GEMMs ALREADY emit `cp.async` (27 groups at num_stages 2, 39 at 3,
  51 at 4, 63 at 5; 20 KB of shared memory per stage of the 227 KB per SM),
- both attention kernels already emit `cp.async` (39 / 78 with the prefix loop),
- the small fused kernels already emit `ld.global.v4`, not scalar `b16`,
- `tl.range(num_stages=)` adds nothing the kernel argument does not already do.
So there is no unexploited pipelining to switch on, and the plan's 20-28%
estimate does not exist. What DOES exist is the depth knob itself, which the
engine set to 2 everywhere on the belief that it was inert: candidate 96 gives
`exact`/`trans`/`hoist` three stages and leaves `gemm` at two so warmup times
them against each other.
The one surviving piece of the original reading: a masked load whose bound the
compiler cannot prove aligned stays scalar. Every hot loop in the engine either
is unmasked or has a constexpr/divisible bound, so there is nothing to collect.
