#!/bin/sh
# Every kernel executed on the CPU by the Triton interpreter (about one minute). Exit status 1 on any FAIL/ERROR/assert.
# Image: docker build -t fasty-tritoninterp:3.5.0 -f Dockerfile.arm64 .
cd "$(dirname "$0")"; status=0
./run.sh run_spec.py 60 || status=1
./run.sh run_attention_twin.py > /dev/null || { echo "FAIL run_attention_twin.py" >&2; status=1; }
./run.sh run_numeric.py | tee /dev/stderr | grep -qE "^(FAIL|ERROR)" && status=1
exit $status
