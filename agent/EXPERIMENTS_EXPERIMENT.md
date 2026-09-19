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

## E02 — single-pass or split dense verification attention

Hypothesis: the block verifier always uses the single-token split-KV policy,
even though one program now handles all draft queries in a KV group. This
can allocate too many partials and use small attention tiles. Add a single-pass
variant which normalizes and writes final outputs directly when no split is
needed, eliminating the second kernel. Compare with the existing split path
and larger tiles during warmup, using random finite Q/K/V and the same tree
mask. All valid keys remain present. Register agreeing options for whole-pass
refinement. Expected signal: lower public-1/2 TPOT, with public-0 pass time
and pacing also benefiting if the reduced overhead pays.

This candidate stacks on E01 while its official run is queued. Keep the default
as fallback; no change to draft policy or cache ownership. Risks: different
softmax reduction ordering, insufficient occupancy for unsplit attention, and
extra warmup. Verify direct output indexing and empty splits on CPU and compile
both single/split specializations for SM90 before submission.

E02 local validation passed: 10 protocol/client tests, direct and split tree
attention CPU reference comparisons (position zero, empty intervals, poisoned
unused slots), 16 SM90 compilation cases, official archive validation. Both
read-only reviews found no concrete blocker; their probe-coverage comment was
addressed by checking all-full chains, all-alternative trees, and mixed rows
with varying positions. The existing compile_spec.py signature was updated.
GPU correctness and warmup/throughput remain unmeasured.

## Results — E01/E02 keep

Baseline b0e2d76: 1063.550 tokens/s, all gates passed.
E01 416fc34: 1072.484 tokens/s (+0.84%). Public rates 311.2/533.6/3171.9
versus baseline 304.4/521.2/3118.5; all public decode times improved, so keep
despite the sub-1% aggregate delta. Prefill did not improve measurably.
E02 065f18f: 1087.324 tokens/s (+1.38% over E01, +2.24% over baseline).
Public rates 307.4/527.5/3245.9; largest signal public-2 TPOT 4.119 ms versus
4.224 ms. All gates passed, peak memory 16.011 GB, run 663 s. Keep.
Raw reports and TSV are in agent/results/ and ignored agent/results.tsv.

## E03 — downstream consumers merge split projections

Live comparison: SSS 1095.135 versus dryfter 1087.324 (gap 0.72%). Their
origin/main at 80d38ef includes our earlier fused MLP and single-pass attention,
plus merge-free projection consumers, two-context draft table and dynamic block
selection. Their best measured commit is b1ca1cc; later mask-free/transposed
GEMM changes have no reported result yet. Avoid treating unmeasured code as a win.

Port the measured b1ca1cc merged.py, residual-add norm and QK/RoPE consumers.
Adapt our linear and layer plumbing to pass FP32 split partials directly during
verification; keep our own fused gate/up backend. Hypothesis: remove three
merge launches per layer (up to 108 per verifier pass), lowering TPOT without
changing block shapes, drafting, or prefill. Preserve BF16 projection rounding
before norm/residual/rotation. FP32 split-add order changes within the contract.
Full-model correctness still requires our own run of the combined stack.

E03 checks passed: 24 CPU consumer indexing/cast cases, 16 SM90 consumer
specializations, 10 protocol/client tests, archive validator, both independent
read-only reviews. Review confirms Split never reaches the ordinary final-token
slicing path and consumers retain partial allocations through their launches.

## E04 — context-robust successor drafts

The competitor's measured two-context table is a separate, bounded improvement:
their offline report gives 15.3% top-1 / 38.9% top-8 successor prediction versus
12.9% / 34.7% for bare-token context, and 2.0–2.7% fewer passes. Their measured
candidate44/48 stack includes it. This is reported lab evidence, not reproduced
on this machine (which lacks the draft lab and model).

Adopt the two contexts (token alone, token following newline), keep the exact
verifier unchanged. Rank the sum of raw FP32 logits instead of computing two
log-softmax tensors: each normalization is a row constant, so ranks are the
same in real arithmetic. Small floating-point tie differences only affect
draft proposals, never accepted correctness. Request logits_to_keep=1 to avoid
computing the unused prefix-token LM head. Expected signal: TPOT at batches
with tree alternatives. Prefill and graph architecture unchanged. Check table
chunk boundaries and ranking equivalence with a deterministic CPU fake model.

E04 checks passed: real successor_table builder tested on CPU fake model for
eight vocabulary/chunk combinations, including partial final chunks and both
contexts; score difference from summed log probabilities is a row constant.
All 10 protocol/client tests and archive validation passed. No new Triton code.

## E05 — even K partitions in the existing skinny GEMM (held)

Competitor's unmeasured exact/transposed kernels suggest a narrower experiment:
offer an evenly dividing split count for the existing 64x128 GEMM tile, alongside
its current power-of-two split count. QKV K=2560 has 20 K tiles: 5 splits remove
4 padded iterations and 37.5% of partials compared with 8 splits. Down K=9728
uses 4 instead of 8 splits, halving partials but possibly reducing occupancy.
Warmup and whole-pass refinement decide, with the old configuration retained.
No new kernel algorithm, operand transpose, mask specialization, or layout
inheritance. Expected effect: incremental TPOT improvement if less partial
traffic pays. Same BF16 boundaries; split reduction ordering can differ.

E05 validation: 85 real-candidate K-coverage cases, four added GEMM SM90
specializations, merged-consumer compile coverage extended to five splits,
32 consumer CPU cases, gated merge BS padding compiled for five splits.
The independent review recommended isolating this partition change from the
competitor's mask-free and transposed kernels. No kernel bodies changed.

## E06 — transposed skinny GEMM to unlock Hopper WGMMA (held)

Offline SM90 inspection of the competing source verifies a concrete codegen
difference: for M16, ordinary/exact tiles emit 16 mma.sync instructions and
zero WGMMA; transposed tiles emit 8 wgmma.mma_async and zero mma.sync. At M32
the ordinary tile has 32 MMA versus 8 WGMMA for transpose. Shared memory stays
20/24 KiB, no PTX local declarations in those tested shapes. These are static
instruction counts, not timing claims.

Implement the transposed operand formulation in our existing linear module,
retain all masks and the established [split,M,N] output. Offer it alongside
our E05 even-split ordinary GEMM. No weight relayout, no new output rounding.
This is distinct from E05 and will be held until a queue slot is available.
Unlike the competing emulator, our check must follow the transposed [N,M]
accumulator and actual transposed stores. Expected signal: projection-dominated
TPOT; could lose if memory bandwidth already dominates or stores get worse.

E06 local checks passed: explicit transposed accumulation/store emulation,
eight SM90 specializations (FP32 partial and direct BF16 outputs), 10 existing
tests, archive validator, two independent static reviews. No speed claim yet.

E03 result: 1073.276 tokens/s (-1.29% vs E02 best), all gates passed.
Public rates 304.0/528.3/3219.5 versus E02 307.4/527.5/3245.9; no compensating
public signal. Discard merge-free consumers on our stack. User specifically
asked to prioritize beating the previous benchmark. E04 may finish to isolate
the draft table's effect; cancel queued E05 as superseded, then test Hopper
with two-context drafts and even splits but without E03 merge consumers.

E05 queued run a4346a68 canceled before execution. E06 rollback restores
decode.py, layers.py, qk_rope.py and rmsnorm.py exactly from best E02 and removes
Split plumbing/unused engine module. Current runtime diff vs best E02 consists
only of linear.py (even splits + Hopper option) and speculate.py (two-context
table). Standard tests and official archive validation pass after rollback.


## E07: two-stage exact vocabulary argmax

Hypothesis: replacing the generic 151936-column indexed reduction with
a block reduction and a small final reduction lowers verify-pass latency,
visible in public TPOT. Ported the independent a787855 reduction; LM-head
projection and BF16 rounding are unchanged. Also used for prefill.
60 CPU exact-index cases cover within/across-block ties, infinity, all
negative/infinite logits, final-entry maxima and ragged tails. Six SM90
variants compiled; 10 protocol tests and archive validation passed.
No measured speed claim; comparison base is E06.


## E08: prioritize repeated layer traffic during refinement (held)

Hypothesis: the 12-second whole-graph refinement budget currently visits the
389M-weight LM head ahead of 36 repeated 50M-weight gate/up projections.
Normalize the vocabulary head priority by 36, leaving every kernel and
choice unchanged. The gated-projection knob keeps its existing priority
just ahead of its fallback projection. This ordering also appears in the
competitor newer unmeasured code; it is not a proven gain.
10 tests, archive validation and diff checks passed. Hold locally while
E06 and E07 occupy the two queued slots.

## E04 failure investigation and rollback

Run e43aec8c-a2aa-40e0-a51d-6dc28e8405aa (1868a441) failed
incorrect_output at 17:06 UTC. All three public cases passed correctness:
317.162 / 530.331 / 3203.409 TPS. No ranked score. Hidden token details
and engine logs are withheld by the service. No timeout or infrastructure error.
The only change versus passing E03 was the two-context successor table;
E03 itself contained the subsequently removed merge consumers. This does
not prove the table caused incorrect output: full verification is unchanged,
and different drafts can expose a preexisting verifier or numerical issue.
400 CPU tree cases (959 alternative branches) passed; no reproducible
bookkeeping bug found. Restored speculate.py exactly from best E02.
Canceled queued E07 e3c3a171 (inherited the unvalidated table). Keep active
E06 as evidence: it has the new table but excludes E03 consumers. New
submission keeps Hopper, argmax and refinement ordering with the validated
bare-token table. Protocol tests, restored builder checks and archive pass.
This is a risk rollback, not a claim to have identified or fixed root cause.

## E06 measured: discard combined candidate; isolate argmax on E02
Run23b323bf (b13cbd2) passed at1084.436 vs E02 best1087.324 (-0.27%).
Public313.663/530.929/3156.675; batch16 down2.75%, TPOT4.279 vs4.119ms.
Aggregate difference alone is within expected noise; no demonstrated net gain.
This tested Hopper/even splits/two-context drafts together, so cannot blame
one component. Passing without E03 consumers narrows but does not resolve
E04's hidden correctness failure. Canceled queued recovery ae73f83.
Restored linear.py exactly from E02, removing Hopper/even splits and untested
refinement priority. Current engine differs from E02 ONLY in argmax kernel
and its three call sites. Next official run isolates token selection.
Existing E05/E06 CPU scripts are historical and require their commit's kernels.
10 protocol tests, archive validator and diff checks passed; argmax kernel
unchanged from its 60 CPU cases/six SM90 compilations/two reviews.

## E10 isolated prefill backend experiment
Compare cuDNN vs existing Flash GQA in causal prefill. Based on rival aeadef2,
with stricter selection: matching tensor strides, three numerical probe scales,
CUDA graph timing, capture on actual caller storage, Flash timing recheck,
5% speed threshold, permanent Flash fallback. Zero-dropout causal calls only.
This is an operator screen; official GPU validation still decides correctness.
Two reviews completed: architecture review no concrete blocker; CLI review
raised GPU validation limitations and probe/key concerns. Addressed causal/
dropout guard, multiple probes, actual-buffer capture. Existing masked paths
are unchanged. CPU tests validate selection control flow, not GPU numerics.
10 protocol tests, six backend cases, cache/capture guard, archive pass.
ISOLATED against E02: only engine/attention.py differs from065f18f.
Argmax remains in its separately queued3d9b781; not discarded before measurement.

## Architecture prototype: fused QK/RoPE/cache + unsplit tree attention
Unsubmitted source and builder under agent/lab. One CTA owns each row/KV head;
history reads strictly precede position; fresh K/V computed from packed QKV
and consumed locally, then uniquely stored. No cross-CTA dependence. Uses
existing tree masks and all BF16 rounding boundaries. Four SM90 compilations
(T2/4/8/16) pass; shared22/22/28/32KiB, no PTX .local declarations. This is
not a runtime register-spill or performance measurement. Next: independent
line-by-line pointer/cast/cache emulation, adversarial review, full-model
validation and bounded optional selection alongside split attention.
Leader now Silver Bullet1103.988; SSS1097.653 (new6334ff5, +0.23% over prior,
817s whole run). Their marginal gain alone does not establish a superior path.

## E11: adopt complete measured SSS engine as competitive baseline
SSS c096f57 scored1129.718 (+3.90% over our1087.324); Silver Bullet best
384edb3 scored1103.988 and its own notes attribute difference from SSS's
1097.7 equivalent baseline to run noise. Silver's refine-before-compare trial
scored1102.318 and was discarded. Latest tip changes are not measured gains.
SSS measured stack combines adaptive block sizing, layout inheritance and
verify-focused tuning, merged consumers, stale-prediction siblings, two-context
table, argmax, optional cuDNN and GC control. Cannot isolate each contribution
from bundled results. Our previous cherry-picks did not reproduce the stack.
E09 isolated argmax1066.895; E10 isolated prefill1066.929: discard both.
E11 engine tree matches c096f57 EXACTLY (git diff --cached c096f57 -- engine
empty). No unrelated competitor files imported. Preserve our history/notes.
400 tree cases pass (1506 stale hints, 961 branches), nine SM90 speculation
kernels compile, 10 protocol tests and archive validator pass. Subagent review
attempt failed on usage limit; standalone Codex review requested separately.
This is baseline reproduction, not an original optimization or guaranteed win.
Fused-attention prototype deferred: rival independently reports excessive
compile/warmup cost and limited application to unsplit attention. Next advance
should start from this measured integrated baseline if our run confirms it.
Standalone Codex review completed: exact snapshot verified, no concrete regression in block selection, stale reset or generator GC finalization.
