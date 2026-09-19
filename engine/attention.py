"""Native Flash GQA prefill and a grouped single-token SDPA reference path."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward


_PREFILL_BACKENDS = {}


def _attend(backend, query, key, value, dropout, scaling, causal):
    with sdpa_kernel(backend):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, scale=scaling, is_causal=causal, enable_gqa=True,
        )


def _prefill_backend(query, key, value, scaling):
    """FLASH, or cuDNN's fused attention where it is usable and measurably faster for this shape.

    Both compute exact attention; only the summation order differs. PyTorch
    2.5.1 ranks cuDNN last after stride and GQA issues, so it must earn its
    place here: once per shape, in eager warmup, on random tensors with the
    real sizes and strides, it has to run, agree with FLASH and be faster.
    """
    shape = (query.device, tuple(query.shape), tuple(key.shape), key.stride(), query.stride())
    if shape not in _PREFILL_BACKENDS:
        if torch.cuda.is_current_stream_capturing():
            return SDPBackend.FLASH_ATTENTION
        _PREFILL_BACKENDS[shape] = SDPBackend.FLASH_ATTENTION
        try:
            generator = torch.Generator(device=query.device).manual_seed(31415)
            probes = [
                torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device).normal_(generator=generator)
                for t in (query, key, value)
            ]
            timings = {}
            for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
                for _ in range(3):
                    out = _attend(backend, *probes, 0.0, scaling, True)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    out = _attend(backend, *probes, 0.0, scaling, True)
                end.record()
                end.synchronize()
                timings[backend] = (start.elapsed_time(end), out)
            flash_ms, reference = timings[SDPBackend.FLASH_ATTENTION]
            cudnn_ms, candidate = timings[SDPBackend.CUDNN_ATTENTION]
            agrees = bool(torch.isfinite(candidate).all()) and float((candidate.float() - reference.float()).abs().max()) <= 0.03
            if agrees and cudnn_ms < 0.95 * flash_ms:
                _PREFILL_BACKENDS[shape] = SDPBackend.CUDNN_ATTENTION
            print(f"prefill attention warmup: flash_ms={flash_ms / 5:.3f} cudnn_ms={cudnn_ms / 5:.3f} agrees={agrees}", flush=True)
        except Exception as error:
            print(f"prefill attention warmup: cuDNN attention unavailable: {error!r}", flush=True)
    return _PREFILL_BACKENDS[shape]


def grouped_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, last_query=False, **kwargs):
    if attention_mask is None:
        assert query.shape[2] == key.shape[2] or (last_query and query.shape[2] == 1)
        # PyTorch 2.5.1's CUDA Flash backend supports GQA and noncontiguous
        # outer strides. Avoid the HF adapter's repeated KV tensors and its
        # three contiguous copies. The last query may read the full prefix
        # noncausally because no key is later than its absolute position.
        causal = query.shape[2] > 1
        backend = _prefill_backend(query, key, value, scaling) if causal else SDPBackend.FLASH_ATTENTION
        output = _attend(backend, query, key, value, dropout, scaling, causal)
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
