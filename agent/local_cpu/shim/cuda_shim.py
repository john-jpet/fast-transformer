"""``torch.cuda`` stand-ins so the engine's CUDA host code runs eagerly on CPU.

A fake CUDA graph cannot record kernels, so ``torch.cuda.graph(g)`` finds the
source of the ``with`` body it guards (AST of the calling function), lets the
body run once eagerly as capture does, and ``g.replay()`` re-executes that body
with the caller's globals and the locals as they were when capture ended.
``torch.cuda.is_current_stream_capturing()`` is True inside capture AND replay,
so "must finish during eager warmup" guards fire exactly as on the GPU.
"""

import ast
import inspect
import random
import sys
import textwrap

import torch

_CAPTURING = [0]
_BODIES = {}
_RNG = random.Random(0)
TIMING = {"mode": "const"}


def _with_body(frame):
    code = frame.f_code
    key = (code, frame.f_lineno)
    if key not in _BODIES:
        lines, first = inspect.getsourcelines(code)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        wanted = frame.f_lineno - first + 1
        nodes = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.With) and n.lineno <= wanted <= n.items[-1].context_expr.end_lineno
        ]
        assert nodes, f"cuda shim: no with-statement at {code.co_filename}:{frame.f_lineno}"
        module = ast.Module(body=nodes[-1].body, type_ignores=[])
        ast.increment_lineno(module, first - 1)
        _BODIES[key] = compile(module, code.co_filename, "exec")
    return _BODIES[key]


class CUDAGraph:
    def __init__(self):
        self.body = self.frame_globals = self.frame_locals = None
        self.replays = 0

    def replay(self):
        assert self.body is not None, "replay of a graph that was never captured"
        _CAPTURING[0] += 1
        try:
            exec(self.body, self.frame_globals, dict(self.frame_locals))
        finally:
            _CAPTURING[0] -= 1
        self.replays += 1

    def reset(self):
        self.body = None


class graph:
    def __init__(self, cuda_graph, pool=None, stream=None, **_):
        self.cuda_graph = cuda_graph

    def __enter__(self):
        frame = sys._getframe(1)
        self.frame = frame
        self.cuda_graph.body = _with_body(frame)
        self.cuda_graph.frame_globals = frame.f_globals
        _CAPTURING[0] += 1

    def __exit__(self, *error):
        _CAPTURING[0] -= 1
        self.cuda_graph.frame_locals = dict(self.frame.f_locals)
        self.frame = None
        return False


class Event:
    def __init__(self, enable_timing=False, **_):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def synchronize(self):
        assert self.recorded, "synchronize() on an event that was never recorded"

    def query(self):
        return True

    def elapsed_time(self, end):
        assert self.recorded and end.recorded
        return _RNG.uniform(0.5, 1.5) if TIMING["mode"] == "random" else 1.0


class Stream:
    def __init__(self, device=None, **_):
        pass

    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class stream:
    def __init__(self, s):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *error):
        return False


def _cpu(device):
    if isinstance(device, str) and device.startswith("cuda"):
        return "cpu"
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.device("cpu")
    return device


def random_time():
    return _RNG.uniform(0.5, 1.5) if TIMING["mode"] == "random" else 1.0


def install(timing="const", seed=0):
    TIMING["mode"] = timing
    _RNG.seed(seed)
    cuda = torch.cuda
    cuda.CUDAGraph, cuda.graph, cuda.Event, cuda.Stream, cuda.stream = CUDAGraph, graph, Event, Stream, stream
    cuda.current_stream = lambda device=None: Stream()
    cuda.synchronize = lambda device=None: None
    cuda.is_current_stream_capturing = lambda: _CAPTURING[0] > 0
    cuda.mem_get_info = lambda device=None: (40 << 30, 80 << 30)
    cuda.empty_cache = lambda: None
    cuda.max_memory_allocated = lambda device=None: 0
    cuda.reset_peak_memory_stats = lambda device=None: None

    module_to = torch.nn.Module.to

    def to(self, *args, **kwargs):
        args = tuple(_cpu(a) for a in args)
        kwargs = {k: _cpu(v) for k, v in kwargs.items()}
        return module_to(self, *args, **kwargs)

    torch.nn.Module.to = to
