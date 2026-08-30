"""Per-arm summary of Stage-1 rollouts: stop reasons, turns, commit behaviour, decision-tag counts."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
phase = sys.argv[1] if len(sys.argv) > 1 else "dt_capture"
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == phase and r.get("status") == "ok"]
by = collections.defaultdict(list)
for r in rows:
    by[r["condition"]].append(r)
for cond, rs in sorted(by.items()):
    stops = collections.Counter(r.get("stop_reason") for r in rs)
    turns = sorted(r.get("n_turns", 0) for r in rs)
    sc = sum(str((r.get("judge") or {}).get("is_shortcut")).lower() == "true" for r in rs)
    tags = collections.Counter()
    n_with = collections.Counter()
    for r in rs:
        for t, k in (r.get("dt_tag_counts") or {}).items():
            tags[t] += k
            if k:
                n_with[t] += 1
    att = blk = suc = 0
    for r in rs:
        p = run / r["transcript_path"]
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        cs = [x for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit")]
        att += bool(cs); blk += any(x.get("exit_code", 0) != 0 for x in cs); suc += any(x.get("exit_code", 0) == 0 for x in cs)
    print(f"{cond}: n={len(rs)} shortcuts={sc} stop={dict(stops)} turns median={turns[len(turns) // 2]} min={turns[0]} max={turns[-1]}")
    print(f"   commit attempted/blocked/succeeded = {att}/{blk}/{suc}")
    print("   tag totals: " + ", ".join(f"{t}={tags[t]} (in {n_with[t]} rollouts)" for t in sorted(tags)))
    wall = sorted(r.get("wall_s", 0) for r in rs)
    print(f"   wall median={wall[len(wall) // 2]:.0f}s max={wall[-1]:.0f}s")
