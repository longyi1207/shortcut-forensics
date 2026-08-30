"""Stage 3a analysis — rank full-attention heads by how much they READ the
instruction span at decision tokens (from scripts/dt_heads.py traces).

Unit of analysis = rollout (per-rollout means, then mean/SE across rollouts).
For every (layer, head):
  dec_instr   mean mass on the instruction span at decision steps (commit_cmd U post_fail_first20)
  ctl_instr   same at control steps (first_tok of non-failed turns U other_cmd)
  dec_ctrl    mass on the length-matched task-text span at decision steps
Ranking score = dec_instr (what is actually read at the decision), with
dec_instr - dec_ctrl reported as the instruction-specificity check and
dec_instr - ctl_instr as the decision-specificity check.
Writes outputs/<run>/dt_heads_rank.json with:
  table, top8/top16 (as {layer: [heads]}), rand8_s0/rand8_s1 (size-matched random
  head sets, fixed seeds), layer_totals; these feed dt_mask.py via SCFX_DTM_HEADS.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from scripts.run_phase import read_jsonl

run = Path("outputs/20260821-launch")
PHASE = "dt_heads"
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("trace_path")]
by_cond = {}
for r in rows:
    by_cond.setdefault(r["condition"], []).append(r)
print({c: len(v) for c, v in by_cond.items()})

DEC_TAGS = {1, 2}      # post_fail_first20, commit_cmd
CTL_TAGS = {0, 3}      # first_tok, other_cmd
S = {"instr": 0, "ctrl": 1, "notes": 2, "recent": 3}


def per_rollout(r):
    z = np.load(run / r["trace_path"])
    tag, prev_rc, mass = z["tag"], z["prev_rc"], z["mass"]  # mass [n, Lf, H, 4]
    if mass.shape[0] == 0:
        return None
    dec = np.isin(tag, list(DEC_TAGS))
    ctl = np.isin(tag, list(CTL_TAGS)) & ~((tag == 0) & (prev_rc > 0))
    out = {"full_layers": z["full_layers"].tolist(), "n_dec": int(dec.sum()), "n_ctl": int(ctl.sum())}
    for name, m in (("dec", dec), ("ctl", ctl)):
        out[name] = mass[m].mean(0) if m.any() else None  # [Lf, H, 4]
    return out


def stack(cond, key, span):
    vals = []
    for r in by_cond.get(cond, []):
        p = per_rollout(r)
        if p and p[key] is not None:
            vals.append(p[key][:, :, S[span]])
    return np.stack(vals) if vals else None  # [R, Lf, H]


for cond in by_cond:
    print(f"\n== {cond}: rollouts={len(by_cond[cond])}")
    for key in ("dec", "ctl"):
        for span in S:
            a = stack(cond, key, span)
            if a is None:
                continue
            fl = per_rollout(by_cond[cond][0])["full_layers"]
            lt = a.mean(0).sum(1)  # per-layer total mass (sum over heads) [Lf]
            print(f"  {key}/{span:6s} R={a.shape[0]:2d}  per-layer mass(sum heads): " + " ".join(f"L{L}={v:.2f}" for L, v in zip(fl, lt)))

cond = "dth_prompt"
dec_i, ctl_i, dec_c = stack(cond, "dec", "instr"), stack(cond, "ctl", "instr"), stack(cond, "dec", "ctrl")
if dec_i is None:
    raise SystemExit("no dth_prompt traces with decision steps yet")
fl = per_rollout(by_cond[cond][0])["full_layers"]
R, Lf, H = dec_i.shape
mean_dec, se_dec = dec_i.mean(0), dec_i.std(0, ddof=1) / np.sqrt(R) if R > 1 else np.zeros_like(dec_i.mean(0))
table = []
for li in range(Lf):
    for h in range(H):
        row = {"layer": int(fl[li]), "head": h, "dec_instr": float(mean_dec[li, h]), "se": float(se_dec[li, h]),
               "ctl_instr": float(ctl_i[:, li, h].mean()) if ctl_i is not None else None,
               "dec_ctrl": float(dec_c[:, li, h].mean()) if dec_c is not None else None}
        row["spec_instr"] = row["dec_instr"] - row["dec_ctrl"] if row["dec_ctrl"] is not None else None
        row["spec_dec"] = row["dec_instr"] - row["ctl_instr"] if row["ctl_instr"] is not None else None
        table.append(row)
table.sort(key=lambda r: -r["dec_instr"])
print(f"\nTop 20 heads by instruction mass at decision steps (R={R} rollouts; uniform would be ~{1 / 1000:.4f} per key*span):")
for r in table[:20]:
    print(f"  L{r['layer']:2d} h{r['head']:2d}  dec_instr={r['dec_instr']:.3f}±{r['se']:.3f}  ctl_instr={r['ctl_instr'] if r['ctl_instr'] is None else round(r['ctl_instr'], 3)}  dec_ctrl={r['dec_ctrl'] if r['dec_ctrl'] is None else round(r['dec_ctrl'], 3)}")
tot = sum(r["dec_instr"] for r in table)
print(f"total instr mass summed over all {len(table)} heads at decision steps = {tot:.2f}; top8 share = {sum(r['dec_instr'] for r in table[:8]) / max(tot, 1e-9):.2f}, top16 share = {sum(r['dec_instr'] for r in table[:16]) / max(tot, 1e-9):.2f}")


def as_set(rows_):
    d = {}
    for r in rows_:
        d.setdefault(str(r["layer"]), []).append(int(r["head"]))
    return {k: sorted(v) for k, v in d.items()}


rng0, rng1 = np.random.default_rng(0), np.random.default_rng(1)
all_pairs = [(int(fl[li]), h) for li in range(Lf) for h in range(H)]
top8, top16 = table[:8], table[:16]
top_keys8 = {(r["layer"], r["head"]) for r in top8}
top_keys16 = {(r["layer"], r["head"]) for r in top16}
rest8 = [p for p in all_pairs if p not in top_keys8]
rest16 = [p for p in all_pairs if p not in top_keys16]
rand8_s0 = [dict(layer=L, head=h) for L, h in [rest8[i] for i in rng0.choice(len(rest8), 8, replace=False)]]
rand8_s1 = [dict(layer=L, head=h) for L, h in [rest8[i] for i in rng1.choice(len(rest8), 8, replace=False)]]
rand16_s0 = [dict(layer=L, head=h) for L, h in [rest16[i] for i in rng0.choice(len(rest16), 16, replace=False)]]
out = {"n_rollouts": R, "full_layers": fl, "n_heads": H, "table": table,
       "top8": as_set(top8), "top16": as_set(top16), "rand8_s0": as_set(rand8_s0), "rand8_s1": as_set(rand8_s1), "rand16_s0": as_set(rand16_s0),
       "layer_totals_dec_instr": {str(int(fl[li])): float(mean_dec[li].sum()) for li in range(Lf)}}
(run / "dt_heads_rank.json").write_text(json.dumps(out, indent=1))
print("wrote", run / "dt_heads_rank.json", "| top8 =", out["top8"], "| rand8_s0 =", out["rand8_s0"])
