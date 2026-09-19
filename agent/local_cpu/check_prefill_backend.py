"""Exercise backend choice/fallback with real CPU tensors and mocked GPU calls."""
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engine'))
import torch
import attention as a

torch.set_num_threads(1)
flash, cudnn = a.SDPBackend.FLASH_ATTENTION, a.SDPBackend.CUDNN_ATTENTION
query = torch.zeros(2, 8, 17, 16).transpose(1, 2).contiguous().transpose(1, 2)
key = torch.zeros(2, 2, 25, 16)[:, :, :17]
value = torch.zeros_like(key)
for mode, expected in [('fast', cudnn), ('slow', flash), ('wrong', flash), ('unsupported', flash), ('capture_error', flash), ('clock_ramp', flash)]:
    a._PREFILL_BACKENDS.clear()
    calls = []
    def attend(backend, q, k, v, dropout, scaling, causal):
        assert causal and dropout == 0 and q.stride() == query.stride()
        assert k.stride() == key.stride() and v.stride() == value.stride()
        calls.append(backend)
        if mode == 'unsupported' and backend == cudnn:
            raise RuntimeError('not supported')
        return torch.full_like(q, 1 if mode == 'wrong' and backend == cudnn else 0)
    flash_timings = [0]
    def timing(fn):
        fn()
        backend = calls[-1]
        if mode == 'capture_error' and backend == cudnn:
            raise RuntimeError('graph capture unavailable')
        if backend == flash:
            flash_timings[0] += 1
            return 0.5 if mode == 'clock_ramp' and flash_timings[0] > 1 else 1.0
        return 1.1 if mode == 'slow' else 0.8
    with patch.object(a, '_attend', attend), patch.object(a, '_graph_time', timing), patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
        assert a._prefill_backend(query, key, value, 0.25) == expected
        before = len(calls)
        assert a._prefill_backend(query, key, value, 0.25) == expected
        assert len(calls) == before
a._PREFILL_BACKENDS.clear()
with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True), patch.object(a, '_attend', side_effect=AssertionError('must not tune inside capture')):
    assert a._prefill_backend(query, key, value, 0.25) == flash
    assert not a._PREFILL_BACKENDS
print('prefill backend choice: six selection/fallback cases, cached choices and capture guard passed')
