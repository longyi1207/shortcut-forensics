"""Validate decision-token capture traces: dt_ arrays present per decision layer,
tag counts, vector shapes, and that commit_cmd/post_fail tags appear when the
transcript has the corresponding events. Usage: dt_check.py <phase> [n_last]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from scripts.run_phase import read_jsonl
from src.proj_trace import DT_TAGS

run = Path("outputs/20260821-launch")
phase = sys.argv[1] if len(sys.argv) > 1 else "dt_smoke"
n_last = int(sys.argv[2]) if len(sys.argv) > 2 else 2
inv = {v: k for k, v in DT_TAGS.items()}
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == phase and r.get("trace_path")]
if not rows:
    raise SystemExit(f"no rows for phase {phase}")
ok_all = True
for r in rows[-n_last:]:
    z = np.load(run / r["trace_path"])
    t = json.loads((run / r["transcript_path"]).read_text())
    n_commit = sum(1 for x in t if x.get("role") == "tool" and str(x.get("command", "")).lstrip().startswith("git commit"))
    n_fail = sum(1 for x in t if x.get("role") == "tool" and x.get("exit_code", 0) != 0)
    print(f"record {r['id']} {r['condition']} status={r['status']} turns={r.get('n_turns')} tags={r.get('dt_tag_counts')} | transcript: git-commit cmds={n_commit} failed tools={n_fail}")
    layers = r.get("decision_layers") or []
    for L in layers:
        p = f"L{L}_dt_"
        if p + "vecs" not in z.files:
            print(f"  L{L}: MISSING dt arrays"); ok_all = False; continue
        V, tag, step, turn = z[p + "vecs"], z[p + "tag"], z[p + "step"], z[p + "turn"]
        counts = {inv[int(k)]: int(v) for k, v in zip(*np.unique(tag, return_counts=True))} if tag.size else {}
        norms = np.linalg.norm(V.astype(np.float32), axis=1) if V.size else np.array([])
        print(f"  L{L}: vecs={V.shape} {V.dtype} counts={counts} mean||v||={norms.mean() if norms.size else float('nan'):.1f} turns={sorted(set(turn.tolist()))[:6]}...")
        ok_all &= V.ndim == 2 and V.shape[0] == tag.size == step.size == turn.size and (V.shape[1] > 1000 if V.size else True)
    # consistency: if the transcript has a git commit, commit_cmd should be present at least once
    if n_commit > 0:
        has = any(np.isin(z[f"L{L}_dt_tag"], [DT_TAGS["commit_cmd"]]).any() for L in layers if f"L{L}_dt_tag" in z.files)
        print(f"  commit_cmd captured: {has}"); ok_all &= has
    if n_fail > 0:
        has = any(np.isin(z[f"L{L}_dt_tag"], [DT_TAGS["post_fail_first20"]]).any() for L in layers if f"L{L}_dt_tag" in z.files)
        print(f"  post_fail_first20 captured: {has}"); ok_all &= has
print("DT_CAPTURE_OK" if ok_all else "DT_CAPTURE_FAIL")
