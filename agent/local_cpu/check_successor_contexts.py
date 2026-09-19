"""Exercise the real table builder on CPU without loading model weights."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine"))
import torch
from speculate import successor_table


class Model:
    def __init__(self, vocabulary):
        self.weight = torch.empty(vocabulary, 1)
        self.calls = []

    def get_input_embeddings(self):
        return self

    def scores(self, ids):
        columns = torch.arange(self.weight.shape[0])[None, :]
        center = (ids[:, -1:] * 7 + (11 if ids.shape[1] == 2 else 0)) % self.weight.shape[0]
        return (-(columns - center).abs().float() + columns.float() / 1024).bfloat16()

    def __call__(self, input_ids, use_cache, logits_to_keep=None):
        assert not use_cache
        assert input_ids.shape[1] in (1, 2)
        if input_ids.shape[1] == 2:
            assert (input_ids[:, 0] == 198).all()
        self.calls.append(input_ids.clone())
        return SimpleNamespace(logits=self.scores(input_ids)[:, None, :])


for vocabulary in (17, 257):
    for chunk in (1, 7, 64, 2048):
        model = Model(vocabulary)
        got = successor_table(model, chunk=chunk)
        ids = torch.arange(vocabulary)[:, None]
        bare = model.scores(ids).float()
        after = model.scores(torch.cat((torch.full_like(ids, 198), ids), 1)).float()
        expected = bare.topk(8, dim=-1).indices
        assert torch.equal(got, expected) and got.shape == (vocabulary, 8)
        assert got.is_contiguous() and got.dtype == torch.int64
        assert len(model.calls) == (vocabulary + chunk - 1) // chunk
        assert all(ids.shape[1] == 1 for ids in model.calls)
print("restored bare-token successor builder passed: chunk tails, shapes, ranking identity")
