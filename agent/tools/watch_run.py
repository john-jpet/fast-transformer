"""Wait for the official run of a commit; save the raw result; print a summary. Never starts runs."""
import json, os, sys, time
from dryft_api import get
sha = sys.argv[1]; deadline = time.time() + 4 * 3600
run_id = None
def safe(path):
    try: return get(path)
    except Exception as e: return {'_exc': repr(e)}
while time.time() < deadline and run_id is None:
    runs = safe('/api/v1/runs'); items = runs.get('items', []) if isinstance(runs, dict) else []
    for it in items or []:
        if str(it.get('commitSha', '')).startswith(sha):
            run_id = it['id']; print('run', run_id, it.get('state'), flush=True); break
    if run_id is None: time.sleep(15)
if run_id is None: print('NO RUN FOUND before deadline'); sys.exit(2)
while time.time() < deadline:
    r = safe(f'/api/v1/runs/{run_id}'); run = r.get('run', r)
    if run.get('state') in ('succeeded', 'failed', 'timed_out', 'canceled', 'infra_error'):
        json.dump(run, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', f'{run_id}.json'), 'w'), indent=1)
        res = run.get('result') or {}
        print('RUN', run_id, run.get('state'), 'ranked', res.get('ranked'), 'score', res.get('score'), 'fail', res.get('failureCode'), res.get('failureMessage'), run.get('errorCode'), run.get('errorMessage'))
        print('peakMemoryGB', round(((res.get('metrics') or {}).get('peakMemoryBytes') or 0)/1e9, 3), 'duration_s', run.get('startedAt'), run.get('finishedAt'))
        for s in res.get('shapes', []):
            m = s.get('modelMetrics') or {}
            print(s.get('id'), s.get('caseStatus'), 'tps %.1f total %.1f ttft %.2f (nat %.1f) tpot %.3f (nat %.2f) %s' % (s.get('tokensPerSecond') or 0, s.get('p50Ms') or 0, m.get('ttftMs') or 0, m.get('referenceTtftMs') or 0, m.get('tpotMs') or 0, m.get('referenceTpotMs') or 0, s.get('caseMessage')), 'p10/p50/p90 %.1f/%.1f/%.1f sd %.2f' % (s.get('p10Ms') or 0, s.get('p50Ms') or 0, s.get('p90Ms') or 0, s.get('stddevMs') or 0))
        sys.exit(0)
    time.sleep(20)
print('TIMEOUT waiting; run', run_id, 'not terminal'); sys.exit(3)
