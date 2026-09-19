"""Save every finished run that is not in agent/results yet, then rebuild results.tsv. Read-only on the platform."""
import json, os, subprocess, sys
from dryft_api import get
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for item in get("/api/v1/runs").get("items", []):
    path = os.path.join(root, "results", f"{item['id']}.json")
    if item.get("state") in ("succeeded", "failed", "timed_out", "canceled", "infra_error") and not os.path.exists(path):
        run = get(f"/api/v1/runs/{item['id']}"); run = run.get("run", run)
        json.dump(run, open(path, "w"), indent=1)
        result = run.get("result") or {}
        print(item["commitSha"][:7], run.get("state"), result.get("score"), result.get("failureCode"),
              [round(s.get("tokensPerSecond") or 0, 1) for s in result.get("shapes", [])])
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_results_tsv.py")])
