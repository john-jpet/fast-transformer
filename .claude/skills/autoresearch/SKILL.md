---
name: autoresearch
description: Autonomous experiment loop in the style of karpathy/autoresearch, adapted to the Dryft Kernel Rush Qwen3 engine (fasty-autoreg-transformer) - propose one idea, implement it, check it without a GPU, push (= one official H100 run), log keep/discard/crash in results.tsv, advance or revert, and never stop to ask. Use when asked to "autoresearch", "keep cooking", loop on the engine score, or run experiments unattended.
---

# autoresearch — Kernel Rush edition

The LLM does its own research. Adapted from https://github.com/karpathy/autoresearch (`program.md`):
same loop, different lab. There the editable file is `train.py`, the fixed judge is `prepare.py`,
the metric is `val_bpb` after 5 minutes. Here:

| autoresearch | this repo |
| --- | --- |
| `train.py` (the only file you edit) | everything under `engine/` (the only thing submitted) |
| `prepare.py` (read-only judge) | the Dryft platform: a push to `main` = one official H100 run (~8-10 min, FIFO queue, 15-minute hard cap, engine stdout hidden) |
| `val_bpb` (lower is better) | `result.score`, tokens/s, geometric mean of six hidden workloads (HIGHER is better); plus per-public-case tokens/s, TTFT, TPOT, p10/p50/p90 |
| `results.tsv` | `agent/results.tsv` (untracked) + prose in `agent/EXPERIMENTS.md` |
| `uv run train.py > run.log` | `git push origin HEAD`, then `cd agent/tools && nohup python3 watch_run.py <sha> > /tmp/watch_<sha>.log 2>&1 &` |

## Setup (once per session)

1. Read `agent/continue.md` (state, best commit, risks), the tail of `agent/EXPERIMENTS.md`, and
   `agent/results.tsv`. Read `AGENTS.md` invariants once.
2. Check the queue and the board: `cd agent/tools && python3 -c "from dryft_api import get; print(get('/api/v1/runs')['items'][:3]); print(get('/api/v1/challenges/decode/leaderboard')['items'][:3])"`.
   Never print `.env`.
3. Go. No confirmation needed: the user pre-authorized commits, pushes and official runs.

## What you CAN / CANNOT do

- CAN: change anything under `engine/` that keeps output EXACTLY the model's greedy tokens
  (layouts, CUDA graphs, fused Triton kernels, exact speculative decoding, pacing).
- CANNOT: quantize/approximate, ship weights or non-Triton GPU code, use the network in the engine,
  carry token state across generations, probe hidden workloads, print `.env`, reset the dirty tree,
  stage `agent/ATTACK_PLAN.md` / `agent/CLAUDE_SYSTEM_PROMPT.md`, or spend money (e.g. Modal) without asking.
- Simplicity criterion (from autoresearch): a small gain that adds ugly complexity is not worth
  keeping; deleting code for equal score is a win. Identical code varies about ±1% in score, so a
  change under ~1% needs a second run or a public-case signal before it counts as `keep`.

## The experiment loop — LOOP FOREVER

1. Pick ONE idea. Write the hypothesis in `agent/EXPERIMENTS.md`: expected effect, which public
   number will show it (public-0 = batch 1, public-1 = batch 4 x 2048 prompt, public-2 = batch 16 x 128 outputs).
2. If the idea is about speculative drafts, rank it OFFLINE first in `~/.cache/fasty-lab`
   (pinned model on MPS, saved greedy samples, simulators, `REPORT.md`); kill it there if it is < 2%.
3. Implement. No-GPU checks, all must pass: `python3 -m unittest discover -s tests`;
   `set -a; . ./.env; set +a; ./bin/dryft validate engine`; new Triton kernels compile for H100
   offline (`agent/local_cpu/offline_compile.py` inside Docker image `fasty-cpucheck:3.1.0`; the
   Triton interpreter does NOT work there); new index/mask/bookkeeping formulas are emulated line by
   line in plain Python against a reference (`agent/local_cpu/check_*.py`).
4. `git add <explicit paths> && git commit && git push origin HEAD` — this IS the experiment run.
5. Start the background watcher and IMMEDIATELY begin the next idea; do not wait on results. But the
   platform runs ONE submission at a time (~10 min): keep AT MOST TWO runs queued. Finished work
   beyond that is committed locally and pushed when a slot frees up, bundled deliberately (a stacked
   candidate inherits its predecessors). A long queue makes the submissions page unreadable for the
   user and delays every result by an hour.
6. Collect results without blocking: `cd agent/tools && python3 collect_runs.py` saves every finished
   run to `agent/results/` and rebuilds `agent/results.tsv`. The platform queue is FIFO per team (about
   10 minutes per run): keep at most ~4 runs queued and drop superseded ones with
   `./bin/dryft cancel <run_id>`. Each row of `agent/results.tsv` (tab-separated) is:
   `commit	score	public0_tps	public1_tps	public2_tps	status	description` with status
   `keep` (new best or clear public-case win), `discard`, or `crash` (failed/canceled run; score 0).
7. `keep` -> the branch advances. `discard` -> undo it with a NEW commit that restores the previous
   behaviour (never `git reset` pushed history; the leaderboard keeps the best run anyway).
   `crash` -> read `failureCode`/`failureMessage`; fix something dumb once, otherwise skip the idea.
   A lone `incorrect_output` on an untouched path has happened as a rare flake: rerun once.
8. Update `agent/continue.md` every few experiments so a fresh session can take over.

**Parallel help (bounded, read-only, written deliverables):** adversarial reviews of every kernel
change by a Claude subagent AND `codex exec -s read-only "<prompt>" < /dev/null > out.txt 2>&1`
(it hangs without `< /dev/null`); web research scouts; lab analysts.

**NEVER STOP.** Do not ask "should I keep going?". The human may be asleep and expects results when
they return. If you run out of ideas, think harder: reread `EXPERIMENTS.md` dead ends, mine the lab
report, combine near-misses, try a more radical change. You stop only when manually stopped.
