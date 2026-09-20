"""Pack projections that share an input, without changing their BF16 outputs."""

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from attention import grouped_sdpa
from kernels.decode_attention import block_attention, decode_attention
from kernels.linear import MAX_ROWS, linear
from kernels.qk_rope import qk_rope_cache
from kernels.swiglu import swiglu


class PackedAttention(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.head_dim = reference.head_dim
        self.layer_idx = reference.layer_idx
        self.scaling = reference.scaling
        self.is_causal = reference.is_causal
        self.num_key_value_groups = reference.num_key_value_groups
        self.q_width = reference.q_proj.out_features
        self.kv_width = reference.k_proj.out_features
        self.qkv_weight = torch.nn.Parameter(
            torch.cat((reference.q_proj.weight, reference.k_proj.weight, reference.v_proj.weight), dim=0),
            requires_grad=False,
        )
        self.q_norm = reference.q_norm
        self.k_norm = reference.k_norm
        self.o_proj = reference.o_proj
        self.train(reference.training)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, last_token_only=False, **kwargs):
        input_shape = hidden_states.shape[:-1]
        output_shape = (input_shape[0], 1) if last_token_only else input_shape
        assert not last_token_only or (past_key_value is not None and past_key_value.prefilling)
        head_shape = (*input_shape, -1, self.head_dim)
        # In a verify block the consumer kernels take a split projection's
        # FP32 partials directly (kernels/merged.py): no merge launch.
        block = past_key_value is not None and not past_key_value.prefilling and hidden_states.shape[1] > 1
        # One-token decode steps (batches without speculation) hand their split
        # partials to the same consumers: 144 merge launches fewer per step.
        split = past_key_value is not None and not past_key_value.prefilling
        packed = linear(hidden_states, self.qkv_weight, split_ok=split)
        # Two entries: gathered cos/sin. Three: the whole tables plus each
        # block token's RoPE phase (verify blocks): the kernel reads them in place.
        cos, sin, *phases = position_embeddings
        if past_key_value is not None:
            key = past_key_value.keys[self.layer_idx]
            value = past_key_value.values[self.layer_idx]
            query = qk_rope_cache(
                packed, self.q_norm, self.k_norm, cos, sin, cache_position,
                key, value, self.q_width // self.head_dim,
                prefill=past_key_value.prefilling, rows=block, phases=phases[0] if phases else None,
            )
            if last_token_only:
                # The final prompt query follows every cached key, so it is
                # exactly one dense decode read ending at its own position.
                attention = decode_attention(
                    query[:, :, -1:, :].contiguous(), key, value, cache_position[-1:], self.scaling,
                )
            elif block:
                # Verify blocks: row b's tokens sit at cache_position[b] + t.
                # The fused kernel stored Q token-major, as the kernel expects.
                attention = block_attention(
                    query.transpose(1, 2), key, value, cache_position, self.scaling,
                    chain=past_key_value.chain,
                )
            elif past_key_value.prefilling:
                length = hidden_states.shape[1]
                attention, _ = grouped_sdpa(
                    self, query, key[:, :, :length, :], value[:, :, :length, :],
                    attention_mask, scaling=self.scaling, dropout=0.0,
                )
            else:
                attention = decode_attention(query, key, value, cache_position, self.scaling)
        else:
            q, k, v = packed.split((self.q_width, self.kv_width, self.kv_width), dim=-1)
            query = self.q_norm(q.reshape(head_shape)).transpose(1, 2)
            key = self.k_norm(k.reshape(head_shape)).transpose(1, 2)
            value = v.reshape(head_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            attention, _ = grouped_sdpa(
                self, query, key, value, attention_mask, scaling=self.scaling, dropout=0.0
            )
        return linear(attention.reshape(*output_shape, -1).contiguous(), self.o_proj.weight, split_ok=split), None


#: Prefill MLP row-block. 1024 rows keep the gate/up tile at 39.8 MB, inside a
#: 50 MB L2; 2048 would be 79.7 MB and miss it, 512 doubles the launches for no
#: further benefit. Decode never reaches it -- a verify block is at most 64 rows.
PREFILL_CHUNK = 1024


class PackedMLP(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.gate_up_weight = torch.nn.Parameter(
            torch.cat((reference.gate_proj.weight, reference.up_proj.weight), dim=0),
            requires_grad=False,
        )
        self.down_proj = reference.down_proj
        self.train(reference.training)

    def forward(self, hidden_states, split_ok=False):
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if not split_ok and rows > PREFILL_CHUNK:
            return self.chunked(hidden_states, rows)
        gate_up = linear(hidden_states, self.gate_up_weight, split_ok=split_ok)
        return linear(swiglu(gate_up), self.down_proj.weight, split_ok=split_ok)

    def chunked(self, hidden_states, rows):
        """The prefill MLP a row-block at a time, so the gate/up tile stays in L2.

        At a 2048-token prompt the gate/up output is [8192, 19456] BF16, 318.8
        MB: cuBLAS writes it to HBM and ``swiglu`` reads every byte straight
        back, because nothing that large lives in a 50 MB L2. A 1024-row block
        makes it 39.8 MB, which does, so the write is absorbed and the read
        never leaves the cache -- about 23 GB across the 36 layers, near 6.8 ms
        of bus time. Eight blocks a layer cost 21 extra launches at ~1.1 us, so
        the trade nets about +6 ms of a 120 ms prefill.

        Rows of a GEMM are independent and SwiGLU is elementwise, so every value
        is the one the single call produced. The block count is fixed by the
        prompt shape, so the captured graph stays static.
        """
        width = hidden_states.shape[-1]
        flat = hidden_states.reshape(rows, width)
        out = torch.empty((rows, width), dtype=hidden_states.dtype, device=hidden_states.device)
        for start in range(0, rows, PREFILL_CHUNK):
            block = flat[start:start + PREFILL_CHUNK]
            gate_up = linear(block, self.gate_up_weight)
            out[start:start + PREFILL_CHUNK] = linear(swiglu(gate_up), self.down_proj.weight)
        return out.reshape(hidden_states.shape)
