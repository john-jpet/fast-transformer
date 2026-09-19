#!/bin/sh
# usage: ./run.sh run_spec.py [args]  -- one script in the Triton interpreter against this checkout's engine/ (read-only mounts)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
exec docker run --rm -v "$REPO/engine":/work/engine:ro -v "$REPO/agent/local_cpu":/emu:ro -v "$HERE":/lab:ro \
  -e PYTHONPATH=/work/engine:/lab:/emu -e TRITON_INTERPRET=1 -e FASTY_STOCK=${FASTY_STOCK:-0} fasty-tritoninterp:3.5.0 python "/lab/$1" $2 $3 $4
