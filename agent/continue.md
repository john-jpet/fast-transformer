# Latest update — 2026-09-19 21:50 UTC

Best SSS run: **1144.3**, normalized **1143.7**, c91 `c758faf`, 692 s.
The newer c93 `44edf62` scored 1103.3 / normalized 1111.5 in 833 s;
the extra refinement budget has not established a gain. Keep the best tree.

Current experiments:
- c94 `9e15bd4`, direct NumPy completion-stamp reads: SSS run
  `384e1371-67ac-48a7-b460-e23c1762880a`; also dryfter `61cef2a`.
  All local gates passed; expect negligible score impact, useful as a repeat
  of c93's tuning behavior. CPU forced-usable alias/reset/fallback checks passed.
- c95 `0fbb6d7`, TMA weight loads for fused lm_head/argmax, dispatched to
  Silver Bullet `77f25a5`; isolated worktree `/tmp/fasty-c95`. All local gates,
  H100 compilation, and descriptor-coordinate emulation passed. GPU descriptor
  behavior remains to be judged. This variant is not included in c96.
- c96, incremental sibling membership in `_propose`: identical drafts in 300
  interpreter cases; all gates pass. New code is six lines; PTX shrank 74% at
  T=16 and shared memory 1024 -> 32 bytes. Performance remains unmeasured.

Next promising work: refine() must compare an incumbent and challenger freshly,
not a fresh challenger to a possibly 20-second-old minimum. Hold the incumbent
CUDA graph and restore it when rejecting a challenger to avoid a recapture.
This was independently identified by the Claude Code adversarial review.

Public archive upload currently returns HTTP 405; use the authorized GitHub
push workflow. The documentation-only run `21cb72e2` was canceled before start.
Do not push documentation alone: every main push triggers an official run.
Read `agent/EXPERIMENTS.md` for exact validation and dispatch records.

The older notes below are historical where they disagree with this update.

# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~17:50 UTC)

**Leaderboard #1: 1130.6 tokens/s** (commit `822ce98`, candidate 67: TMA GEMM kind +
warmup bundle; whole run 612 s against the 900 s cap; previous best c57 1129.7). Silver Bullet 1112.7 (fork
`sivakovivan/silver-transformer`, tracks our main, consented to idea sharing;
their one idea - refine each block size before comparing - is in c60), dryfter
1087.3 (`john-jpet/fast-transformer`), zip 1059.4. User target: 1200.
Progress today: c48 1095.1 -> c53 1097.7 (GEMM tiles, keep_native) -> c54
1115.0 (two-stage Triton argmax + trimmed budgets) -> c57 1129.7 (stale-guess
sibling + cuDNN prefill option + frozen GC).
History since c57: c58 (32 MiB cuBLAS workspace) 1111.9 discard; c62 1090.2
(slow node; batch-1 pass slower: the mask-free attention loops cost occupancy
at batch 1); c66 1112.9 (warmup bundle: run 815 -> 763 s); **c67 `822ce98`
1130.6 BEST, 612 s** (TMA descriptor-load GEMM kind + refine on the three
fastest layouts + runtime COUNT + shared-newline successor table; batch-4 TPOT
-7%, batch-16 -4%).
**BEST: 1140.0 (candidate 85, `53e4a7a`).** Silver Bullet's copy of that tree
drew 1137.7 and dryfter's dispatch drew 1136.5, so the level is real and all
three queues are now ours. Leaderboard: SSS 1140.0, Silver Bullet 1137.7,
dryfter 1136.5, zip 1059.4.
c85 = PDL off + fused lm_head/argmax knob + pinned-memory completion stamps +
in-place RoPE tables, on the TMA GEMM kinds. 648 s of the 900 s limit.
EVERYTHING SINCE HAS BEEN 1.2-1.6% BELOW IT and has been bisected down:
c86 (plain-decode attention search above batch 16) reverted; c89's third pass
in flight reverted (it delayed the next sample's prefill: public-1/2 TPOT rose
1-2%); c87 (pacing pass time = fastest of five back-to-back groups) and c88
(embedding gather fused into the first norm) measured neutral and stay.
Measuring: c91 `c758faf` (warp-width knobs for 17-64-row blocks and the
embedding norm). Queued: c93 `44edf62` (the lookahead revert + refinement
budget 16 -> 24 s). **If c93 does not come back to ~1135 normalized, the next
step is to diff `44edf62` against `53e4a7a` and drop c87/c88 too - i.e. return
to the exact 1140 tree and rebuild from there one change per run.**
MERGED TEAM: dryfter (`john-jpet/fast-transformer`) and Silver Bullet
(`sivakovivan/silver-transformer`) are extra run queues; we have push access.
Dispatch without rewriting their history:
`git fetch <url> +main:refs/remotes/X/main && git push <url> $(git commit-tree
<our-sha>^{tree} -p <our-sha> -p X/main -m "...") :refs/heads/main`.
Their results are visible ONLY as their leaderboard best.
Dead offline today (no runs spent): alignment hints (no PTX change at all),
mask-free RMSNorm rewrite (doubles reads per row).
READ RESULTS WITH `cd agent/tools && python3 collect_runs.py | tail -1 && python3 report_runs.py`:
one line per run with duration, the node-speed control (native prefill TTFT),
the NORMALIZED score and public TTFT/TPOT probes. Normalized, c57 1129.7, c58
1119.9, c62 1107.0, c66 1119.8, and dryfter's copy of c57 scored 1123.9: the
c57-class engine is ~1120-1125 and c57 itself was a good draw. Only changes of
>= 2% normalized are readable in one run.
RULES OF THE ROAD: keep one run measuring + one queued; record run duration
(`finishedAt - startedAt`) with every score - c52 was canceled at 917 s; every
warmup second costs six. The harness hides engine stdout on purpose (hidden
shapes could be encoded in text): do NOT build side channels (e.g. through
peak memory) - refused once already.
LOCAL GATES before every push (all exist, ~3 min total):
`python3 -m unittest discover -s tests`; `./bin/dryft validate engine`;
`agent/local_cpu/interp/all.sh` (every kernel executed on CPU by a patched
Triton interpreter); `~/.cache/fasty-lab/venv/bin/python
agent/local_cpu/smoke_engine.py` (whole engine, real host control flow, 4-layer
real model vs HF greedy); the offline cuda:90 compile script of any new kernel.
PLAN + research: `agent/NEXT_PLAN.md`, 15 reports in `~/.cache/fasty-lab/plan/`,
paste-able research prompt `agent/RESEARCH_PROMPT.md`.
Key findings: at batch 1 the median sample sits on the pacing floor (0.70 x
pass time) so pass-TIME cuts convert 1:1; history-copy drafting is at its
ceiling (a perfect source selector would save only 3-5% of passes); dead in the
lab today: depth-2 trees, logit re-ranking, hidden-state match selection.
Parked on worktree branches: speculation for batches 17-64
(`worktree-agent-a8f491703758a9453`, ~0 expected gain), fused QK-norm/RoPE/KV +
attention kernel (`worktree-agent-ad7f188c606e1cd22`, bit-exact but compiles
25-35x slower than the two kernels it replaces).
Known bad: cuBLASLt preference, forcing 64-row blocks, ranked-draft kernel
(c49), paired gate/up block kernel (c50).

## What produced the jump from 933

Exact self-speculative decoding (`engine/speculate.py`, `engine/kernels/spec.py`,
block kernels in `kernels/decode_attention.py` and `kernels/qk_rope.py`):
drafts copied from each row's own history (3/2/1-token suffix match) or a
model-derived successor table; one graphed verify pass per block; per-row
positions; rows never pass their last requested token; tokens released no
faster than `PACE` x pass time to bound the 25% spread gate.
c20 948 (batch 1) -> c22 985 (batched) -> c23 1003 (fused bookkeeping) ->
c25 1042 (1-token matches) -> c27 1051 (tree drafts) -> c28 1061 (32-row GEMM)
-> c29 1065 (adaptive pacing). Losers: c26 long chains / 24-row cuBLAS blocks,
c30 32-row blocks at batch 2. Dead offline: token recycling, model-view drafts,
second-level alternatives, lag-based row dealing, frequency votes.

## Tools that now exist

- **Local draft lab** `~/.cache/fasty-lab` (not in the repo): the pinned model on
  MPS, greedy continuations, successor top-8 table, simulators. Draft policies
  can be ranked offline in seconds; the ranking transferred to the platform.
- `agent/local_cpu/`: offline `cuda:90` Triton compilation, CPU emulations of
  every speculation kernel formula (`check_tree.py`, `check_block_attention.py`,
  `check_spec_kernels.py`, `test_speculate.py`, `test_spec_queue.py`).
- Reviews: Claude subagents and `codex exec -s read-only` both work well.

## Known risks

- A rare `incorrect_output` occurred once on the plain batch-16 path (c22 first
  run; identical engine passed on rerun). Failed runs do not replace the best.
- 15-minute whole-run cap; keep warmup lean.
- `PACE` (0.75) binds at batch 1; lowering it trades speed for spread risk.

## Do not

- Reset the dirty tree, reveal `.env`, start duplicate runs, or touch
  `agent/ATTACK_PLAN.md` / `agent/CLAUDE_SYSTEM_PROMPT.md` (untracked, not ours).
