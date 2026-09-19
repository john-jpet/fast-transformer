"""Fake ``triton.language``: only names evaluated at import time need to exist."""


class constexpr:  # used as an annotation and nothing else
    def __init__(self, value=None):
        self.value = value


def __getattr__(name):  # tl.float32, tl.int64, ... inside never-run kernel bodies
    if name.startswith("__"):
        raise AttributeError(name)
    return name
