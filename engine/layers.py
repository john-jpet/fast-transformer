"""Pack projections that share an input, without changing their BF16 outputs."""

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from attention import grouped_sdpa
from kernels.decode_attention import block_attention, decode_attention
from kernels.gated_linear import gated_linear
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
        packed = linear(hidden_states, self.qkv_weight, split_ok=block)
        cos, sin = position_embeddings
        if past_key_value is not None:
            key = past_key_value.keys[self.layer_idx]
            value = past_key_value.values[self.layer_idx]
            query = qk_rope_cache(
                packed, self.q_norm, self.k_norm, cos, sin, cache_position,
                key, value, self.q_width // self.head_dim,
                prefill=past_key_value.prefilling, rows=block,
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
        return linear(attention.reshape(*output_shape, -1).contiguous(), self.o_proj.weight, split_ok=block), None


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
        if rows > MAX_ROWS and hidden_states.dtype == torch.bfloat16:
            # Prefill: gate/up GEMM with the SwiGLU epilogue, if it measured faster.
            return linear(gated_linear(hidden_states, self.gate_up_weight), self.down_proj.weight)
        gate_up = linear(hidden_states, self.gate_up_weight, split_ok=split_ok)
        return linear(swiglu(gate_up), self.down_proj.weight, split_ok=split_ok)
