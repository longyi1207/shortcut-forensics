"""Re-judge rollouts of a given phase whose judge verdict is missing (Azure 429
casualties), appending to the sidecar outputs/<run>/rejudge.jsonl (idempotent:
skips ids already in the sidecar). Usage: python scripts/rejudge_phase.py <phase> [<phase> ...]"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl
from src import judge as judge_mod
from src.agent_loop import transcript_to_text
from src.clock import append_jsonl

run = Path("outputs/20260821-launch")
phases = set(sys.argv[1:]) or {"b5_prompt_decay"}
side = run / "rejudge.jsonl"
done = {r["id"] for r in read_jsonl(side)} if side.exists() else set()
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") in phases and r.get("status") == "ok" and r.get("transcript_path")]
todo = [r for r in rows if not (isinstance(r.get("judge"), dict) and r["judge"].get("is_shortcut") is not None) and r["id"] not in done]
print(f"phases={sorted(phases)} rows={len(rows)} to re-judge={len(todo)}", flush=True)
for r in todo:
    t = json.loads((run / r["transcript_path"]).read_text())
    dpath = run / "diffs" / r["id"] / "diff.txt"
    diff = dpath.read_text() if dpath.exists() else ""
    j = None
    for attempt in range(4):
        j = judge_mod.judge_rollout(transcript_to_text(t), diff, errors_log=run / "incidents" / "judge_errors.jsonl", rollout_id=r["id"] + "_rejudge")
        if isinstance(j, dict) and j.get("is_shortcut") is not None:
            break
        time.sleep(30)
    append_jsonl(side, {"id": r["id"], "phase": r["phase"], "condition": r["condition"], "judge": j})
    print(r["id"], r["condition"], "->", None if not j else j.get("is_shortcut"), flush=True)
    time.sleep(8)
print("done", flush=True)
