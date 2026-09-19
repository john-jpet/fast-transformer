"""CPU smoke test of the WHOLE engine (no CUDA, no Triton): host-code crashes,
yield counts and greedy exactness, before spending a remote H100 run.

    cd ~/.cache/fasty-lab && . venv/bin/activate
    python <repo>/agent/local_cpu/smoke_engine.py            # first 4 layers, ~2 min
    python <repo>/agent/local_cpu/smoke_engine.py --layers 0 # full model, slow

Real: engine.py, decode.py (DecodeState, capture/choose_block/refine, host pass
queue, pacing, absorb/fill), layers.py, attention.py, tune.py and every kernel
WRAPPER (layout choice, Split hand-off, asserts). Replaced: Triton kernel bodies
(shim/references.py, by kernel name), CUDA graphs (shim/cuda_shim.py re-executes
the captured with-body), events/streams (no-ops), kernel timing helpers, and the
successor table (drafts only; a cheap synthetic one unless --real-table).
Exit status: 0 clean (near-ties within the judge's 2.0 margin are reported),
1 on an exception, a wrong yield count/shape, or a token outside the margin.
"""

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "shim"), os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine")]

import torch  # noqa: E402
# Before the shims: Dynamo validates torch.cuda.Event's base class at import,
# and nothing in Torch/Transformers may mistake the fake triton for a real one.
_path = [entry for entry in sys.path if entry != os.path.join(HERE, "shim")]
_saved, sys.path[:] = sys.path[:], _path
import torch._dynamo  # noqa: E402,F401
import torch.utils._triton  # noqa: E402
torch.utils._triton.has_triton_package()
torch.utils._triton.has_triton()
import transformers.models.qwen3.modeling_qwen3  # noqa: E402,F401
import transformers.integrations.sdpa_attention  # noqa: E402,F401
sys.path[:] = _saved

import cuda_shim  # noqa: E402

TEXTS = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox "
    "jumps over the lazy cat, and then the quick brown fox jumps over the lazy dog again and again and again.",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n\n"
    "def div(a, b):\n    return a / b\n\ndef mod(a, b):\n    return a % b\n",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,",
    "In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded "
    "as a bad move. Many were increasingly of the opinion that they had all made a big mistake in coming down.",
]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/.cache/fasty-lab/model"))
    parser.add_argument("--layers", type=int, default=4, help="keep the first N layers (0 = full model)")
    parser.add_argument("--shapes", default="1x32x16,2x24x12,1x16x2", help="batch x prompt x output, comma separated")
    parser.add_argument("--timing", default="random", choices=("random", "const"),
                        help="fake kernel/pass timings: random exercises layout switching in tune/refine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-table", action="store_true", help="build the real successor table (very slow on CPU)")
    parser.add_argument("--real-tuning", action="store_true", help="keep the kernels' own timing loops (slow)")
    parser.add_argument("--mutate", choices=("relocate", "limit"), help="break the engine on purpose: the run must FAIL")
    parser.add_argument("--strict", action="store_true", help="fail on near-ties too")
    return parser.parse_args()


def load(path, layers):
    from transformers import AutoModelForCausalLM
    extra = {"num_hidden_layers": layers} if layers else {}
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, **extra
    ).eval()


@torch.inference_mode()
def native_greedy(model, prompts, count):
    """Native cached greedy decoding; end-of-sequence is an ordinary token."""
    ids = torch.tensor(prompts)
    out = model(input_ids=ids, use_cache=True)
    steps = []
    for _ in range(count):
        token = out.logits[:, -1, :].argmax(-1)
        steps.append(token.tolist())
        out = model(input_ids=token[:, None], past_key_values=out.past_key_values, use_cache=True)
    return steps


@torch.inference_mode()
def replay_gaps(model, prompt, produced):
    """The judge's check: teacher-forced gap between the argmax and each produced token."""
    ids = torch.tensor([prompt + produced])
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1].float()
    chosen = logits.gather(1, torch.tensor(produced)[:, None])[:, 0]
    return (logits.max(-1).values - chosen).tolist()


def synthetic_table(vocabulary, sequences, seed):
    """Drafts only. True successors at rank 0 (chains accept), rank 1-2 (alternatives win), or absent."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.randint(0, vocabulary, (vocabulary, 8), generator=generator)
    for sequence in sequences:
        for index, (token, following) in enumerate(zip(sequence, sequence[1:])):
            if index % 4 < 2:
                table[token, index % 4] = following
    return table.contiguous()


def main():
    args = parse()
    shapes = [tuple(int(v) for v in shape.split("x")) for shape in args.shapes.split(",")]
    cuda_shim.install(args.timing, args.seed)
    import triton
    import references  # noqa: F401  registers the kernel references
    import transformers
    from transformers import AutoTokenizer
    transformers.logging.set_verbosity_error()  # truncation leaves checkpoint weights unused

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = [tokenizer(text)["input_ids"] for text in TEXTS]
    started = time.time()
    native = load(args.model, args.layers)
    print(f"native model: {len(native.model.layers)} layers, {time.time() - started:.0f}s", flush=True)

    # Per shape: warmup, two samples on different prompts, then an abandoned
    # generator followed by a full one (state reset), all against native greedy.
    plans = []
    for batch, prompt, output in shapes:
        sets = []
        for variant in range(2):
            rows = [encoded[(variant + row) % len(encoded)] for row in range(batch)]
            rows = [(row * (prompt // len(row) + 1))[:prompt] for row in rows]
            sets.append(rows)
        order = [sets[0], sets[1], sets[0], sets[1]]
        expected = [native_greedy(native, prompts, output) for prompts in sets]
        plans.append(((batch, prompt, output), order, [expected[0], expected[1], expected[0], expected[1]]))
    print(f"native greedy references done, {time.time() - started:.0f}s", flush=True)

    import engine as engine_module
    import decode
    import kernels.decode_attention
    import kernels.linear

    if not args.real_table:
        sequences = [p + [step[i] for step in exp] for _, order, exps in plans for prompts, exp in zip(order, exps) for i, p in enumerate(prompts)]
        engine_module.successor_table = lambda model: synthetic_table(model.config.vocab_size, sequences, args.seed)
    if not args.real_tuning:
        def once(fn, *unused):
            fn()
            return cuda_shim.random_time()
        kernels.linear._cold_graph_time = once
        kernels.decode_attention._graph_time = once
    if args.layers:
        pretrained = engine_module.AutoModelForCausalLM.from_pretrained
        engine_module.AutoModelForCausalLM.from_pretrained = (
            lambda *a, **k: pretrained(*a, num_hidden_layers=args.layers, **k))
    # The CPU has no Flash GQA kernel: let SDPA choose its own backend for prefill.
    import attention
    import contextlib
    attention.sdpa_kernel = lambda *backends: contextlib.nullcontext()
    # A replayed prefill graph contains the kernels recorded with prefilling=True.
    prefill_forward = decode.DecodeState.prefill_forward

    def prefill_as_captured(self):
        previous, self.cache.prefilling = self.cache.prefilling, True
        try:
            prefill_forward(self)
        finally:
            self.cache.prefilling = previous
    decode.DecodeState.prefill_forward = prefill_as_captured

    if args.mutate == "relocate":  # a kept alternative's K/V never moves to its slot
        decode.spec.relocate = lambda *a, **k: None
    if args.mutate == "limit":  # rows run past the last requested token
        settle = decode.spec.settle
        decode.spec.settle = lambda tokens, greedy, position, limit, *rest: settle(tokens, greedy, position, limit + 64, *rest)
    failures, near_ties = [], []
    engine = engine_module.Engine(args.model)
    print(f"engine loaded, {time.time() - started:.0f}s", flush=True)
    for shape, order, expecteds in plans:
        batch, prompt, output = shape
        for number, (prompts, expected) in enumerate(zip(order, expecteds)):
            label = f"shape {shape} generation {number}"
            began = time.time()
            generator = engine.generate(prompts, output)
            if number == 2:
                # Abandoned after three tokens, as a harness timeout would; the next call must not care.
                got = [next(generator) for _ in range(min(3, output))]
                generator.close()
                expected = expected[:len(got)]
            else:
                got = list(generator)
            if len(got) != len(expected):
                failures.append(f"{label}: yielded {len(got)} times, wanted {len(expected)}")
            if any(not isinstance(step, list) or len(step) != batch or any(type(t) is not int for t in step) for step in got):
                failures.append(f"{label}: a step is not a list of {batch} ints")
                continue
            wrong = [(s, r) for s, (a, b) in enumerate(zip(got, expected)) for r in range(batch) if a[r] != b[r]]
            state = engine.state
            detail = f"speculative={state.speculative} block={state.block_size}"
            if state.speculative:
                detail += f" passes={state.passes_enqueued} for {output - 1} tokens"
            print(f"{label}: {len(got)} steps, {len(wrong)} mismatches, {detail}, {time.time() - began:.1f}s", flush=True)
            for row in sorted({r for _, r in wrong}):
                first = min(s for s, r in wrong if r == row)
                produced = [step[row] for step in got]
                gaps = replay_gaps(native, prompts[row], produced)
                message = (f"{label} row {row}: first mismatch at step {first} (engine {produced[first]}, native "
                           f"{expected[first][row]}), teacher-forced gap there {gaps[first]:.3f}, max gap {max(gaps):.3f}")
                (failures if max(gaps) > 2.0 or args.strict else near_ties).append(message)
    print("kernel launches:", dict(sorted(triton.LAUNCHES.items())))
    print("kv relocations (kept alternatives):", references.RELOCATIONS[0])
    for message in near_ties:
        print("NEAR-TIE (inside the 2.0 margin):", message)
    for message in failures:
        print("FAIL:", message)
    print("SMOKE", "FAILED" if failures else "OK", f"{time.time() - started:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        traceback.print_exc()
        print("SMOKE FAILED (exception)")
        status = 1
    sys.exit(status)
