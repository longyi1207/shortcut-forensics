"""Did every rollout in a masking arm actually receive the intervention?

The mask is applied only on turns that follow a failed tool result. A rollout that
never hits one is counted in the treatment arm while having been treated exactly
like a control -- which dilutes the measured effect toward the no-treatment rate.
The gdnswap arm had the same failure mode in a different form (OOM silently
skipping the ablation), so check it here across every masking condition before
the Stage-2 numbers are quoted again.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scipy.stats import fisher_exact

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
side = {r["id"]: r["judge"] for r in read_jsonl(run / "rejudge.jsonl")} if (run / "rejudge.jsonl").exists() else {}
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == "dt_mask" and r.get("status") == "ok"]


def sc(r):
    j = r.get("judge")
    if not (isinstance(j, dict) and j.get("is_shortcut") is not None):
        j = side.get(r["id"])
    return isinstance(j, dict) and str(j.get("is_shortcut")).lower() == "true"


print("{:26s} {:>5s} {:>8s} {:>10s} {:>16s} {:>16s}".format(
    "condition", "n", "untouch", "mean mt", "rate (all)", "rate (treated)"))
out = {}
for cond in sorted({r["condition"] for r in rows}):
    g = [r for r in rows if r["condition"] == cond]
    mt = [len(r.get("masked_turns") or []) for r in g]
    treated = [r for r, m in zip(g, mt) if m > 0]
    untouched = len(g) - len(treated)
    ra = "{}/{}".format(sum(sc(r) for r in g), len(g))
    rt = "{}/{}".format(sum(sc(r) for r in treated), len(treated)) if treated else "-"
    print("{:26s} {:5d} {:8d} {:10.1f} {:>16s} {:>16s}".format(
        cond, len(g), untouched, sum(mt) / max(len(mt), 1), ra, rt))
    if treated:
        out[cond] = (sum(sc(r) for r in treated), len(treated))

print("\n--- treated-only comparisons ---")
if "dtm_prompt_mask_both" in out and "dtm_prompt_mask_toolout" in out:
    (a, na), (b, nb) = out["dtm_prompt_mask_both"], out["dtm_prompt_mask_toolout"]
    print("  mask_both {}/{} vs size-matched toolout control {}/{}  Fisher p={:.3f}".format(
        a, na, b, nb, fisher_exact([[a, na - a], [b, nb - b]])[1]))
    print("  (if the toolout control were also ~25%, the source+copies story would be")
    print("   just 'more context removed is worse'; it needs to stay near the prompt rate)")
