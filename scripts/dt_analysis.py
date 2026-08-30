"""Stage-1 analysis: where does the prompt's effect live at DECISION tokens?

For each decision layer and tag (turn_start, post_fail_start, post_fail_first20,
commit_cmd, other_cmd, every50), compare dt_prompt vs dt_baseline rollouts:
  * per-rollout mean vector per tag -> dh = mean_P - mean_B, ||dh||/||h||,
    cos(dh, tedium_coarse) (L19 direction; at other layers the coarse vector is
    "borrowed", reported for reference only), cos(dh, sae26) at L26
  * tedium projection (coarse@19, sae composite@26) per rollout per tag ->
    Mann-Whitney over rollouts
  * SAE feature diff per tag at layers with an available SAE (19, 26, and any
    full-attention layer for which layer{L}.sae.pt exists): Cohen's d over
    per-rollout means with the active-in->=3-rollouts filter; top features;
    number with |d|>1 vs the every50 control tag
  * DECISION vs CONTROL contrast: is ||dh|| at commit_cmd / post_fail_first20
    larger than at other_cmd / every50 (same rollouts)?
Usage: dt_analysis.py [phase] [cond_base] [cond_prompt]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import mannwhitneyu

from scripts.run_phase import read_jsonl
from src.directions import load_vector
from src.proj_trace import DT_TAGS

run = Path("outputs/20260821-launch")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "dt_capture"
COND_A = sys.argv[2] if len(sys.argv) > 2 else "dt_baseline"
COND_B = sys.argv[3] if len(sys.argv) > 3 else "dt_prompt"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
TOPK = 100
MIN_ROLLOUTS_ACTIVE = 3
TOP_N = 12
TAGS = ["turn_start", "post_fail_start", "post_fail_first20", "commit_cmd", "other_cmd", "every50"]
device = "cuda" if torch.cuda.is_available() else "cpu"
inv = {v: k for k, v in DT_TAGS.items()}

tedium_coarse = load_vector(run / "vectors" / "tedium")["d"].astype(np.float32)
probe26 = json.loads((run / "phase4" / "sae_probe_tedium_L26.json").read_text())
sae26 = np.zeros_like(tedium_coarse)
for i, f in enumerate(probe26["top_features"][:10]):
    sae26 += float(np.sign(f["mean_diff"])) * load_vector(run / "vectors" / f"sae_tedium_top10_L26_f{i}")["d"].astype(np.float32)
TEDIUM_DIR = {19: tedium_coarse, 26: sae26}

rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("trace_path")]
arms = {c: [r for r in rows if r["condition"] == c] for c in (COND_A, COND_B)}
print(f"phase={PHASE} rollouts:", {c: len(v) for c, v in arms.items()})
if min(len(v) for v in arms.values()) < 3:
    raise SystemExit("need >=3 rollouts per arm")
layers = sorted({L for r in rows for L in (r.get("decision_layers") or [])})

# per arm -> per rollout -> per layer -> per tag -> mean vector (and count)
M = {c: [] for c in arms}
for c, rs in arms.items():
    for r in rs:
        z = np.load(run / r["trace_path"])
        d = {}
        for L in layers:
            p = f"L{L}_dt_"
            if p + "vecs" not in z.files or z[p + "vecs"].size == 0:
                continue
            V, tag = z[p + "vecs"].astype(np.float32), z[p + "tag"]
            d[L] = {}
            for tname, tcode in DT_TAGS.items():
                m = tag == tcode
                if m.any():
                    d[L][tname] = (V[m].mean(0), int(m.sum()), V[m])
        M[c].append(d)


def sae_acts(H, W_enc, b_enc):
    pre = H @ W_enc.T + b_enc
    vals, idx = pre.topk(TOPK, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, idx, vals)
    return acts


print("\n=== token counts per tag (mean per rollout) ===")
for c in arms:
    L0 = layers[0]
    cnt = {t: np.mean([d[L0][t][1] if L0 in d and t in d[L0] else 0 for d in M[c]]) for t in TAGS}
    print(f"  {c:12s} " + "  ".join(f"{t}={cnt[t]:.1f}" for t in TAGS))

summary = {}
for L in layers:
    print(f"\n================ L{L} ================")
    sae = None
    try:
        sp = hf_hub_download(SAE_REPO, f"layer{L}.sae.pt")
        s = torch.load(sp, map_location="cpu")
        sae = (s["W_enc"].float().to(device), s["W_dec"].float(), s["b_enc"].float().to(device))
    except Exception as e:  # noqa: BLE001
        print(f"  (no SAE for layer {L}: {str(e)[:60]})")
    for t in TAGS:
        B = [d[L][t] for d in M[COND_A] if L in d and t in d[L]]
        P = [d[L][t] for d in M[COND_B] if L in d and t in d[L]]
        if len(B) < 3 or len(P) < 3:
            print(f"  [{t}] insufficient rollouts ({len(B)},{len(P)})"); continue
        mB, mP = np.stack([b[0] for b in B]), np.stack([p[0] for p in P])
        dh = mP.mean(0) - mB.mean(0)
        hn = float(np.mean([np.linalg.norm(x[0]) for x in B + P]))
        line = f"  [{t:17s}] nB={len(B):2d} nP={len(P):2d} ||dh||/||h||={np.linalg.norm(dh)/hn:.3f} cos(dh,tedium19)={float(dh @ tedium_coarse/(np.linalg.norm(dh)*np.linalg.norm(tedium_coarse)+1e-8)):+.3f}"
        if L in TEDIUM_DIR:
            pB = mB @ TEDIUM_DIR[L]; pP = mP @ TEDIUM_DIR[L]
            line += f" | tedium proj B={pB.mean():7.2f} P={pP.mean():7.2f} MW p={mannwhitneyu(pP, pB).pvalue:.3f}"
        # SAE diff on per-rollout mean activations over the tag's tokens
        if sae is not None:
            W_enc, W_dec, b_enc = sae
            def per_rollout_acts(items):
                out = []
                for _, _, V in items:
                    A = sae_acts(torch.tensor(V, device=device), W_enc, b_enc).cpu().numpy()
                    out.append(A.mean(0))
                return np.stack(out)
            rB, rP = per_rollout_acts(B), per_rollout_acts(P)
            keep = ((rB > 0).sum(0) >= MIN_ROLLOUTS_ACTIVE) | ((rP > 0).sum(0) >= MIN_ROLLOUTS_ACTIVE)
            diff = rP.mean(0) - rB.mean(0)
            pooled = np.sqrt((rP.var(0, ddof=1) + rB.var(0, ddof=1)) / 2) + 1e-6
            dcoh = diff / pooled * keep
            n_big = int((np.abs(dcoh) > 1).sum())
            line += f" | SAE feats>=3roll={int(keep.sum())} |d|>1={n_big}"
            print(line)
            order = np.argsort(-np.abs(dcoh))[:TOP_N]
            for rank, f in enumerate(order):
                if dcoh[f] == 0:
                    break
                pv = mannwhitneyu(rP[:, f], rB[:, f]).pvalue
                dec = W_dec[:, f].numpy()
                print(f"      {rank:2d} feat={int(f):6d} d={dcoh[f]:+5.2f} p={pv:.3f} meanB={rB.mean(0)[f]:.3f} meanP={rP.mean(0)[f]:.3f} active B/P={int((rB[:, f] > 0).sum())}/{int((rP[:, f] > 0).sum())} cos(dec,tedium19)={float(dec @ tedium_coarse/(np.linalg.norm(dec)*np.linalg.norm(tedium_coarse)+1e-8)):+.3f}")
            summary.setdefault(L, {})[t] = {"n": [len(B), len(P)], "dh_ratio": float(np.linalg.norm(dh) / hn), "n_big": n_big,
                                            "top": [{"feat": int(f), "d": float(dcoh[f])} for f in order[:5]]}
        else:
            print(line)
            summary.setdefault(L, {})[t] = {"n": [len(B), len(P)], "dh_ratio": float(np.linalg.norm(dh) / hn)}

print("\n=== decision vs control: ||dh||/||h|| by tag (rows=layers) ===")
print("  L    " + "  ".join(f"{t:>17s}" for t in TAGS))
for L in layers:
    print(f"  {L:<4d} " + "  ".join(f"{summary.get(L, {}).get(t, {}).get('dh_ratio', float('nan')):17.3f}" for t in TAGS))
(run / "phase4" / f"dt_analysis_{PHASE}.json").write_text(json.dumps(summary, indent=2))
print("wrote phase4/dt_analysis_%s.json" % PHASE)
