"""One line per finished official run: score, duration, node-speed control, normalized score, public probes.

The control is native HF's prefill TTFT on public-1/2: GPU-bound and measured
on the same node in the same run (native TPOT is host-bound and useless).
`norm` = score x control / reference control (candidate 57's node).
Usage: python3 report_runs.py [N=12]
"""
import glob, json, math, sys
from datetime import datetime

REFERENCE = (202.3, 192.0)  # candidate 57 (c096f57): native TTFT on public-1 / public-2, ms

rows = []
for path in glob.glob("../results/*.json"):
    if path.endswith(".logs.json"):
        continue
    run = json.load(open(path))
    result = run.get("result") if isinstance(run, dict) else None
    if not result or not result.get("shapes") or not run.get("startedAt") or not run.get("finishedAt"):
        continue
    shapes = {shape["id"]: shape for shape in result["shapes"] if shape.get("modelMetrics")}
    if len(shapes) < 3:
        continue
    seconds = (datetime.fromisoformat(run["finishedAt"]) - datetime.fromisoformat(run["startedAt"])).total_seconds()
    control = [shapes[name]["modelMetrics"]["referenceTtftMs"] for name in ("public-1", "public-2")]
    node = math.sqrt(control[0] / REFERENCE[0] * control[1] / REFERENCE[1])  # > 1: slower node
    score = result.get("score") or 0.0
    probes = " ".join(
        f"{shapes[name]['modelMetrics']['ttftMs']:.1f}/{shapes[name]['modelMetrics']['tpotMs']:.3f}"
        for name in ("public-0", "public-1", "public-2")
    )
    rows.append((run["startedAt"], f"{run['commitSha'][:7]} score {score:7.1f} norm {score * node:7.1f} node {100 * (node - 1):+5.1f}% {seconds:4.0f}s  ttft/tpot {probes}"))
for _, line in sorted(rows)[-int(sys.argv[1]) if len(sys.argv) > 1 else -12:]:
    print(line)
