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
Since c67: c69 canceled at the cap (64-row tiles: reverted); c76 `17e882c`
1118.8 (normalized 1126.2 = level; batch-1 floor lowered by back-to-back pass
timing); c67 RERUN `7cfb7f1` 1115.5 (same code as 1130.6: run noise ~1.3%,
run time 612 vs 709 s); c79 `d25a167` 1114.1 (normalized 1109.4: the
per-batch EXPECTED_PASSES refit hurt hidden shapes while public-1/2 hit their
best: reverted in c82; TMA attention option + persistent TMA GEMM kinds kept).
MEASURING: c80 `2721207` = Programmatic Dependent Launch (kernels/pdl.py:
`griddepcontrol.wait` first in all 25 kernels + patched Triton launcher +
warmup self-test with full fallback; published +4-13% on decode). QUEUED: c82
`f8da493` (+ fused lm_head/argmax refine knob, table reverted). HELD: c83
`b75e3ef` (pinned-memory completion stamps replace event waits; events remain
the fallback). Subagent use is now restricted by the user (research on cheaper
models only when needed); the Exa key file is in the scratchpad (`.exa_key`).
Research in flight (reports land in `~/.cache/fasty-lab/plan/`):
whole_system_review.md, triton31_hopper_features.md (source audit: TMA for
attention K/V tiles, loop prefetch, num_ctas...), web_hopper_triton.md,
exa_hopper_research.md (Exa API key in the scratchpad file `.exa_key`, user
supplied; official Triton docs have "TMA in Gluon" and "Warp-Group MMA"
tutorials). The organisers (Isaac) are raising GPU concurrency: queue waits
should shrink. A self-scheduled cron tick (every 7 min, session-only) drives
the loop; forks `john-jpet/fast-transformer` (dryfter, 1123.9 = our c57) and
`sivakovivan/silver-transformer` copy our main within the hour.
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
