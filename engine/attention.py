"""Native Flash GQA prefill and a grouped single-token SDPA reference path."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def grouped_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, last_query=False, **kwargs):
    if attention_mask is None:
        assert query.shape[2] == key.shape[2] or (last_query and query.shape[2] == 1)
        # PyTorch 2.5.1's CUDA Flash backend supports GQA and noncontiguous
        # outer strides. Avoid the HF adapter's repeated KV tensors and its
        # three contiguous copies. The last query may read the full prefix
        # noncausally because no key is later than its absolute position.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, scale=scaling,
                is_causal=query.shape[2] > 1, enable_gqa=True,
            )
        return output.transpose(1, 2).contiguous(), None
    if query.shape[2] != 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )

    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    # All query heads represent the SAME absolute token position. Treat each
    # group of heads as SDPA's query-row dimension, with one shared K/V head.
    # These rows are heads, not successive time steps: is_causal MUST be False.
    grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
    output = torch.nn.functional.scaled_dot_product_attention(
        grouped_query, key, value,
        attn_mask=attention_mask, dropout_p=dropout,
        is_causal=False, scale=scaling,
    )
    return output.reshape(batch, 1, query_heads, head_dim), None
