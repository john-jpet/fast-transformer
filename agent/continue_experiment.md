# Continue independent autoresearch

Work on local branch `experiment`. User explicitly requested continuous
throughput research, including architectural overhauls, competing with the
other agent on origin/main. Autoresearch skill is `.claude/skills/autoresearch`.

## Repository routing

- Competitor: origin = ShreyShingala/fasty-autoreg-transformer. Fetch only.
- Our benchmark: benchmark = john-jpet/fast-transformer, connected by user.
- Publish with `git push benchmark HEAD:main`. Never push to origin/main.
- Baseline b0e2d76 was pushed to benchmark/main; result not yet read.
- User supplied Dryft credentials in ignored `.env`; API access verified.
  Use agent/tools/dryft_api.py, never print `.env` contents.
- Baseline run 676484f5-cdc8-409e-b53b-7d0d2da4cbb4 passed: 1063.550.
- E01 416fc34 run 61381803-5628-461c-a59f-d900e98d3f81 passed: 1072.484.
- E02 065f18f run 4f21ebd8-a996-42fa-9b0f-9773d036cd3a passed: 1087.324.
  E01 and E02 kept; TSV rebuilt. Best local measured engine = E02.
- Live competitor SSS 1095.135 at b1ca1cc; origin/main now 80d38ef with
  additional unmeasured GEMM changes. Competitor ported our E01/E02 ideas.
- Background watch_run.py processes write agent/results/watch_baseline.log
  and watch_e01.log, plus raw final JSONs. No duplicate runs needed.
- E03 baf1417: run 84039458-8d32-4c78-9b0e-97c6c50dea21 preparing;
  watcher writes watch_e03.log. E04 adds two-context successor table, raw
  logit ranking and last-only LM-head during table construction. CPU checks pass.

## E01 candidate

Fused gate/up projection with SwiGLU epilogue, plus split-K merge activation.
New engine/kernels/gated_linear.py, called by PackedMLP. Warmup selects vs
the existing implementation for each shape. Requires official H100 validation.
Experiment log: agent/EXPERIMENTS_EXPERIMENT.md.

## Local checks

`python -m unittest discover -s tests` and `bin/dryft.exe validate engine`.
WSL Python: /home/johnp/.cache/fasty-cpucheck/bin/python. CPU torch 2.5.1,
Triton 3.1.0, Transformers 4.51.3 installed there. Host Python is 3.12.
Run agent/local_cpu/check_gated_linear.py and compile_gated_linear.py with
WSL paths under /mnt/c/Users/johnp/Documents/!HTN/fasty-autoreg-transformer.
All passed. Independent subagent and codex exec reviews found no blocker.
CPU checks/compilation do not prove numerical correctness or speed on GPU.

Docker Desktop was launched hidden but its API remained unresponsive.
Use WSL rather than waiting on it. Do not put environments in /tmp: WSL
restart removed the first setup there. CLI installed with upstream SHA check.

## Next

E02 stacks single-pass dense tree attention and bounded shape selection on E01.
No drafting changes. CPU direct/split mask tests, SM90 compilation, two reviews,
protocol tests and archive validation passed. No H100 performance claim yet.

E03 ports measured b1ca1cc merge-free consumers (QK and residual-add norm),
adapting linear/layer plumbing while retaining our gated MLP. Up to 108 fewer
merge launches per verification pass. All CPU/SM90/archive/review checks pass.
Next independent idea: their two-context successor table, with raw-logit sum
equivalent to sum log probabilities for ranking; exact verifier unchanged.

Read baseline and E01 results, record metrics in results.tsv, keep/discard
with new commits. Confirm no regressions in prefill, TPOT, warmup or spread.
Explore next substantial exact-compute idea while jobs run; user encourages
overhauls, not just block-size/pacing tweaks. origin/main last at 039a52e.
