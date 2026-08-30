"""Print the recorded mask/KL-probe fields of the most recent dt_mask rollouts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == "dt_mask"]
for r in rows[-n:]:
    kl = r.get("kl_probe") or []
    pf = [e for e in kl if e.get("post_fail")]
    ct = [e for e in kl if not e.get("post_fail")]
    print("{:24s} {} status={} turns={} masked={} applied={} kl={} (pf={},ctl={}) probes={} wall={} short={}".format(
        r.get("condition"), r.get("id"), r.get("status"), r.get("n_turns"), len(r.get("masked_turns") or []),
        r.get("mask_applied_steps"), len(kl), len(pf), len(ct), len(r.get("probe") or []), r.get("wall_s"),
        (r.get("judge") or {}).get("is_shortcut")))
    for tag, ev in (("pf ", pf[:1]), ("ctl", ct[:1])):
        for e in ev:
            print("     ", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "kl_steps"})
