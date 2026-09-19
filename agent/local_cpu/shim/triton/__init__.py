"""Fake ``triton`` for CPU smoke tests: enough for the engine modules to import.

``@triton.jit`` returns a launcher; ``kernel[grid](*args, **kwargs)`` dispatches
by kernel NAME to a pure-PyTorch reference registered in ``REFERENCES`` (see
``../references.py``). Kernel bodies are never executed. Not shipped in engine/.
"""

from . import language  # noqa: F401

__version__ = "0.0.0-cpu-shim"
REFERENCES = {}
LAUNCHES = {}


def cdiv(a, b):
    return -(-a // b)


def next_power_of_2(n):
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


class _Kernel:
    def __init__(self, fn):
        self.fn, self.__name__ = fn, fn.__name__

    def __getitem__(self, grid):
        name = self.__name__

        def launch(*args, **kwargs):
            if name not in REFERENCES:
                raise NotImplementedError(f"cpu shim: no reference registered for Triton kernel {name!r}")
            grid_tuple = grid if isinstance(grid, tuple) else (grid,)
            assert all(isinstance(g, int) and g >= 1 for g in grid_tuple), f"{name}: bad launch grid {grid_tuple}"
            LAUNCHES[name] = LAUNCHES.get(name, 0) + 1
            return REFERENCES[name](grid_tuple, *args, **kwargs)

        return launch

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self.__name__}: a @triton.jit function cannot be called from host code")


def jit(fn=None, **_):
    return _Kernel(fn) if fn is not None else (lambda f: _Kernel(f))


def reference(name):
    def register(fn):
        REFERENCES[name] = fn
        return fn
    return register
