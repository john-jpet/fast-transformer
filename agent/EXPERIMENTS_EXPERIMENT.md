# Independent experiment branch

## E01 — fuse gate/up projection and SwiGLU

Base: `b0e2d76`. Competitor `origin/main` observed at `039a52e`.
The competitor's handoff reports best eligible score 1065.476 tokens/s at
`8a7b9f5`; this checkout cannot yet independently read the platform results.

Hypothesis: computing corresponding gate/up projection tiles together and
applying SwiGLU before storing them eliminates the packed BF16 intermediate
and one launch per layer. The largest expected signal is public-1 TTFT;
public-0/2 TPOT measures the smaller decode benefit. At 8192 prompt rows,
the eliminated gate/up writes and reads total about 23 GB over 36 layers.
This is a traffic estimate, not a speed prediction. cuBLAS efficiency may
outweigh the saved traffic, so retain the existing path as a warmup-selected
fallback. Shape choices freeze before measured generation.

Keep projection outputs, SiLU output, and activation product rounded to BF16
at their original boundaries. Split-K must reduce in FP32 before any cast or
nonlinearity. No change to speculative proposals or acceptance policy.

Validation planned: existing protocol tests, CPU formula/rounding checks,
offline SM90 compilation with Triton 3.1.0, read-only adversarial review,
archive validation, then official end-to-end measurements. The skill's
Mac draft lab does not exist here; this experiment does not change drafting.

Submission prerequisite: the current docs explicitly reject submission on
non-default branch pushes. A separate connected repository is needed to
run this branch without changing the other agent's main. No `.env` or local
CLI was present at session start. No H100 performance claim yet.

User created and connected `john-jpet/fast-transformer`. Remote `benchmark`
targets it; `origin` remains the competitor. Baseline `b0e2d76` was pushed to
`benchmark/main`; subsequent submissions use `git push benchmark HEAD:main`.
The local branch remains `experiment`. Never push these experiments to origin/main.

E01 local validation passed: 10 protocol/client tests, 124 CPU emulation cases,
all 10 new SM90 kernel specializations compiled through cubin with Triton 3.1,
and the official CLI archive validator. No blocker in a bounded subagent
review or separate `codex exec -s read-only` review. Claude CLI is not installed,
so a Codex subagent supplied that independent review. Reviews are saved in
ignored `agent/results/`. Docker Desktop did not become responsive; compilation
used WSL CPU Torch 2.5.1 and Triton 3.1.0 under Python 3.12, not the runtime's
Python 3.11. This is an offline compiler check, not a runtime/H100 test.

Review fixes: release the unused baseline activation tensor during tuning;
prioritize the fused gate/up choice ahead of its generic projection choice.
Both dense and skinny tuners have separate soft 12-second budgets. Full-model
correctness, benefit, and warmup headroom remain for the official run.
