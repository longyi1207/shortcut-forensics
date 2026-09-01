"""What do the judged shortcuts actually consist of? The narrow command regexes
find nothing (-no-verify / -n / chmod -x / rm hook never appear), so the cheat
must be a CONTENT change rather than a bypass. Print each judged shortcut's
verdict text plus the commands that touched the hook, pyproject or type-ignores."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
SUSPECT = re.compile(r"type:\s*ignore|ignore_errors|exclude\s*=|disallow|strict|pre-commit|--no-verify", re.I)
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("status") == "ok"]
n = 0
for r in rows:
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    if not (isinstance(j, dict) and str(j.get("is_shortcut")).lower() == "true"):
        continue
    n += 1
    if n > 8:
        break
    print(f"\n===== {r['id']}  ({r.get('phase')}/{r.get('condition')}) =====")
    for k in ("workaround_type", "reasoning", "explanation", "summary"):
        if j.get(k):
            print(f"  {k}: {str(j[k])[:420]}")
    p = run / (r.get("transcript_path") or "")
    if p.exists():
        cmds = [str(e.get("command", "")) for e in json.loads(p.read_text()) if e.get("role") == "tool"]
        hits = [c for c in cmds if SUSPECT.search(c)]
        print(f"  {len(cmds)} commands, {len(hits)} touching hook/config/type-ignore; last 3 such:")
        for c in hits[-3:]:
            print(f"    - {c[:200]}")
