"""Run the existing interpreter suite in the local WSL CPU environment."""
import builtins
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[2] / "engine")]
import interp_bf16
import numpy2_memory
import torch
torch.set_num_threads(1)
original_open = builtins.open


def local_open(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("/emu/"):
        path = HERE.parent / path.removeprefix("/emu/")
    return original_open(path, *args, **kwargs)


builtins.open = local_open
script = sys.argv.pop(1)
runpy.run_path(str(HERE / script), run_name="__main__")
