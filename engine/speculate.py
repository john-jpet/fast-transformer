"""Exact self-speculation: propose from each row's history, keep what greedy confirms.

Drafts are copied from a row's own tokens after the most recent earlier
occurrence of its current suffix, or follow a table of the model's own greedy
successor of each single token when the row has nothing to copy. The full
model then scores the trusted token and the drafts in one pass. A draft is kept
only if it equals the model's own greedy choice at that position, so the output
is the greedy sequence whatever the drafts were; bad drafts only cost time.
Rows are independent. Everything is fixed-shape tensor code, safe in a CUDA graph.
"""

import torch
from transformers import DynamicCache


def propose(history, position, count, index, successor):
    """``count`` draft tokens per row to follow ``history[b, position[b]]``.

    ``history`` is int64 [B,C]; row b is known up to ``position[b]`` (int64 [B])
    and zero beyond. ``index`` is ``arange(C)``. ``successor`` is int64 [V]: the
    model's own greedy token after each single token, a prompt-independent
    table. Drafts copy the row's text after the latest earlier occurrence of
    its longest suffix (three, two or one token; measured offline, one-token
    matches beat the table) while that text is known, and otherwise follow the
    table from the previous draft.
    """
    size = history.shape[1]
    place = position[:, None]
    last = history.gather(1, place)
    before = history.gather(1, (place - 1).clamp_min(0))
    earlier = history.gather(1, (place - 2).clamp_min(0))
    column = index[None, :]
    one = (history == last) & (column < place)
    two = one & (torch.roll(history, 1, dims=1) == before) & (column >= 1) & (place >= 1)
    three = two & (torch.roll(history, 2, dims=1) == earlier) & (column >= 2) & (place >= 2)
    # Longer suffix first, then the most recent occurrence.
    rank = torch.where(three, column + 2 * size, torch.where(two, column + size, torch.where(one, column, column - size)))
    best = rank.max(dim=1, keepdim=True).values
    found = best >= 0
    start = torch.where(found, best % size, place)
    drafts = []
    previous = last
    for step in range(1, count + 1):
        source = start + step
        copied = history.gather(1, source.clamp_max(size - 1))
        draft = torch.where(found & (source <= place), copied, successor[previous])
        drafts.append(draft)
        previous = draft
    return torch.cat(drafts, dim=1)


def accept(tokens, greedy):
    """Per row, the number of leading drafts the model itself chose (int64 [B]).

    ``tokens[:, 0]`` is trusted and ``tokens[:, 1:]`` are drafts; ``greedy[:, i]``
    is the model's choice after ``tokens[:, :i + 1]``. Draft i+1 stands only if it
    equals ``greedy[:, i]`` and every earlier draft stood.
    """
    agree = (tokens[:, 1:] == greedy[:, :-1]).to(torch.int64)
    return agree.cumprod(1).sum(1)


def advance(position, gained, limit):
    """Tokens each row really gains: a row never moves past ``limit``.

    ``limit`` is the index of the last token the caller asked for, so rows that
    finish early stop growing (and stop writing new KV slots) while slower rows
    catch up. Returns (gained, new position).
    """
    gained = torch.minimum(gained, (limit - position).clamp_min(0))
    return gained, position + gained


def successor_table(model, chunk=2048, top=8, prefix=198):
    """The model's ``top`` likeliest next tokens after each vocabulary token (int64 [V, top]).

    Prompt-independent: computed from the weights once per process, with the
    native forward, before any prompt is seen. It only ever proposes drafts.
    Two contexts are combined by summing log-probabilities: the token alone,
    and the token after a newline (``prefix``, Qwen's id for "\n"). Alone, a
    token sits at position 0 where it doubles as the attention sink; on the
    model's own greedy text the sum predicts the next token 15.3% of the time
    (top-8 38.9%) against 12.9% (34.7%) for the bare token.
    """
    vocabulary = model.get_input_embeddings().weight.shape[0]
    device = model.get_input_embeddings().weight.device
    table = torch.empty((vocabulary, top), dtype=torch.int64, device=device)
    with torch.inference_mode():
        # The newline's keys and values are the same for every token: compute
        # them once and let each chunk attend to them, instead of forwarding
        # the newline 151936 times.
        newline = model(input_ids=torch.tensor([[prefix]], device=device), use_cache=True).past_key_values
        for begin in range(0, vocabulary, chunk):
            ids = torch.arange(begin, min(begin + chunk, vocabulary), device=device)[:, None]
            bare = model(input_ids=ids, use_cache=False).logits[:, -1, :].float().log_softmax(-1)
            shared = DynamicCache()
            for layer in range(len(newline.key_cache)):
                shared.update(
                    newline.key_cache[layer].expand(ids.shape[0], -1, -1, -1),
                    newline.value_cache[layer].expand(ids.shape[0], -1, -1, -1), layer,
                )
            bare += model(input_ids=ids, past_key_values=shared, use_cache=True).logits[:, -1, :].float().log_softmax(-1)
            table[begin:begin + ids.shape[0]] = bare.topk(top, dim=-1).indices
    return table.contiguous()
