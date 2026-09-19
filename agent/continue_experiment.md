# Continue independent autoresearch

Work on local branch `experiment`. User explicitly requested continuous
throughput research, including architectural overhauls, competing with the
other agent on origin/main. Autoresearch skill is `.claude/skills/autoresearch`.

## Repository routing

- Competitor: origin = ShreyShingala/fasty-autoreg-transformer. Fetch only.
- Our benchmark: benchmark = john-jpet/fast-transformer, connected by user.
- Publish with `git push benchmark HEAD:main`. Never push to origin/main.
- Baseline b0e2d76 was pushed to benchmark/main; result not yet read.
- No Dryft token in checkout at start. User asked to create ignored `.env`.
  Use agent/tools/dryft_api.py once present, never print its contents.

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

Read baseline and E01 results, record metrics in results.tsv, keep/discard
with new commits. Confirm no regressions in prefill, TPOT, warmup or spread.
Explore next substantial exact-compute idea while jobs run; user encourages
overhauls, not just block-size/pacing tweaks. origin/main last at 039a52e.
