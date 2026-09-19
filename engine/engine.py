"""BF16 Qwen3 with a reusable KV cache and one CUDA graph per decode shape."""

import gc

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState, optimize_model
from kernels import pdl
from speculate import successor_table

#: Decode steps enqueued beyond the one being read. Bounded, so an abandoned
#: generator leaves little work behind and the launch queue stays shallow.
LOOKAHEAD = 4


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Decided before the first engine kernel exists: every Triton kernel of
        # a pass may then launch programmatically (kernels/pdl.py).
        pdl.self_test("cuda:0")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        # Prompt-independent draft table for exact speculation: the native
        # model's greedy successor of each single token. Drafts never reach the
        # output unless the full model chooses the same token.
        self.model.successor = successor_table(self.model)
        optimize_model(self.model)
        self.state = None
        # torch.cuda.graph runs a full gc.collect() on entry, and warmup enters
        # dozens of captures: freeze the model's object graph out of its reach.
        gc.collect()
        gc.freeze()

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        if not input_ids or not input_ids[0]:
            raise ValueError("generate requires a nonempty batch and prompt")
        shape = (len(input_ids), len(input_ids[0]), max_new_tokens)
        if any(len(row) != shape[1] for row in input_ids):
            raise ValueError("all prompts must have the same length")

        with torch.inference_mode():
            if self.state is None or self.state.shape != shape:
                # The harness reuses a shape after warmup. Bound memory if a
                # local caller changes it rather than retaining many graphs.
                self.state = None
                self.state = DecodeState(self.model, shape)
                # Everything built so far is permanent: keep the collector
                # from ever walking it again.
                gc.collect()
                gc.freeze()
        # A collection inside a ~100 ms generation is a multi-millisecond stall
        # in one sample only: timing noise the spread gate would see.
        collecting = gc.isenabled()
        gc.disable()
        try:
            with torch.inference_mode():
                state = self.state
                prompt = torch.tensor(input_ids, dtype=torch.int64, device=state.device)
                state.prefill(prompt)
                # Keep a few decode steps queued behind the GPU so it never waits
                # for the consumer; never enqueue past the requested output count.
                state.advance(min(max_new_tokens, 1 + LOOKAHEAD))
                tokens = state.read(0)
            yield tokens
            for step in range(1, max_new_tokens):
                with torch.inference_mode():
                    state.advance(min(max_new_tokens, step + 1 + LOOKAHEAD))
                    tokens = state.read(step)
                yield tokens
        finally:
            if collecting:
                gc.enable()
