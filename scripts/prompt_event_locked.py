"""Event-locked prompt-vs-baseline comparison on the b5_full captures.

The uniform 1-in-50 token subsample answered "is there a persistent, diffuse
prompt imprint" (answer at n=8: small, ~10% of ||h||, diffuse, orthogonal to
tedium). If the prompt instead acts at DECISION moments, the place to look is
the turn-start immediately after a temptation event -- and we store EVERY
turn-start residual. Events, from the transcript:
  blocked_commit : the previous tool result was `git commit ...` with exit != 0
  mypy_fail      : the previous tool result was a mypy run with exit != 0
  any_commit     : the previous tool call was any `git commit`
For each event type and layer, gather the turn-start vectors at the turn right
after the event (one per event; multiple per rollout allowed), then:
  * projection onto tedium (coarse at L19; SAE composite at L26), prompt vs
    baseline, Mann-Whitney over events (and over rollout-means)
  * SAE feature diff with a SANE filter: feature must be active in >= 3 rollouts
    of either arm (fixes the single-rollout d=1e5 artifact of the first pass)
  * ||dh|| / ||h|| and cos(dh, tedium) at the event
"""
from __future__ import annotations

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

run = Path("outputs/20260821-launch")
# usage: prompt_event_locked.py [phase] [cond_baseline] [cond_treatment]
PHASE = sys.argv[1] if len(sys.argv) > 1 else "b5_full"
COND_A = sys.argv[2] if len(sys.argv) > 2 else "b5_baseline"
COND_B = sys.argv[3] if len(sys.argv) > 3 else "b5_prompt"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
TOPK = 100
LAYERS = (19, 26)
MIN_ROLLOUTS_ACTIVE = 3
TOP_N = 15
device = "cuda" if torch.cuda.is_available() else "cpu"

tedium_coarse = load_vector(run / "vectors" / "tedium")["d"].astype(np.float32)
probe26 = json.loads((run / "phase4" / "sae_probe_tedium_L26.json").read_text())
sae26 = np.zeros_like(tedium_coarse)
for i, f in enumerate(probe26["top_features"][:10]):
    sae26 += float(np.sign(f["mean_diff"])) * load_vector(run / "vectors" / f"sae_tedium_top10_L26_f{i}")["d"].astype(np.float32)
TEDIUM_DIR = {19: tedium_coarse, 26: sae26}


def events_for(transcript):
    """Return {event_type: [turn indices whose turn-start follows the event]}.
    Assistant turns are numbered in order; a tool entry belongs to the
    preceding assistant turn k, so the NEXT turn-start is k+1."""
    ev = {"blocked_commit": [], "mypy_fail": [], "any_commit": []}
    k = -1
    for x in transcript:
        if x.get("role") == "assistant":
            k += 1
        elif x.get("role") == "tool":
            cmd = str(x.get("command", "")).lstrip()
            rc = x.get("exit_code", 0)
            if cmd.startswith("git commit"):
                ev["any_commit"].append(k + 1)
                if rc != 0:
                    ev["blocked_commit"].append(k + 1)
            if "mypy" in cmd and rc != 0:
                ev["mypy_fail"].append(k + 1)
    return ev


def sae_acts(H, W_enc, b_enc):
    pre = H @ W_enc.T + b_enc
    vals, idx = pre.topk(TOPK, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, idx, vals)
    return acts


rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("trace_path")]
arms = {c: [r for r in rows if r["condition"] == c] for c in (COND_A, COND_B)}
print(f"phase={PHASE} baseline={COND_A} treatment={COND_B} rollouts:", {c: len(v) for c, v in arms.items()})

# gather turn-start vectors per event
data = {L: {c: {e: [] for e in ("blocked_commit", "mypy_fail", "any_commit")} for c in arms} for L in LAYERS}  # list of (rollout_idx, vec)
ev_counts = {c: {e: 0 for e in ("blocked_commit", "mypy_fail", "any_commit")} for c in arms}
ev_rollouts = {c: {e: set() for e in ("blocked_commit", "mypy_fail", "any_commit")} for c in arms}
for c, rs in arms.items():
    for ri, r in enumerate(rs):
        t = json.loads((run / r["transcript_path"]).read_text())
        ev = events_for(t)
        z = np.load(run / r["trace_path"])
        for L in LAYERS:
            p = f"L{L}_"
            V, idx = z[p + "full_vecs"].astype(np.float32), z[p + "full_idx"]
            turn, step = z[p + "turn"][idx], z[p + "step"][idx]
            start_vec = {int(tt): V[j] for j, (tt, ss) in enumerate(zip(turn, step)) if ss == 0}
            for e, turns in ev.items():
                for tt in turns:
                    if tt in start_vec:
                        data[L][c][e].append((ri, start_vec[tt]))
                        if L == 19:
                            ev_counts[c][e] += 1
                            ev_rollouts[c][e].add(ri)
print("events (count / rollouts-with-event):", {c: {e: f"{ev_counts[c][e]}/{len(ev_rollouts[c][e])}" for e in ev_counts[c]} for c in arms})

for L in LAYERS:
    sae_path = hf_hub_download(SAE_REPO, f"layer{L}.sae.pt")
    sae = torch.load(sae_path, map_location="cpu")
    W_enc, W_dec, b_enc = sae["W_enc"].float().to(device), sae["W_dec"].float(), sae["b_enc"].float().to(device)
    for e in ("blocked_commit", "mypy_fail", "any_commit"):
        B, P = data[L][COND_A][e], data[L][COND_B][e]
        print(f"\n=== L{L} event={e}: baseline {len(B)} events/{len({b[0] for b in B})} rollouts, prompt {len(P)} events/{len({p[0] for p in P})} rollouts ===")
        if len(B) < 3 or len(P) < 3:
            print("  insufficient events"); continue
        vB, vP = np.stack([b[1] for b in B]), np.stack([p[1] for p in P])
        pB, pP = vB @ TEDIUM_DIR[L], vP @ TEDIUM_DIR[L]
        # rollout-mean version (independence)
        rmB = [np.mean([pb for (ri, _), pb in zip(B, pB) if ri == k]) for k in sorted({b[0] for b in B})]
        rmP = [np.mean([pp for (ri, _), pp in zip(P, pP) if ri == k]) for k in sorted({p[0] for p in P})]
        print(f"  tedium projection at event turn-start: baseline mean={pB.mean():7.2f} prompt mean={pP.mean():7.2f}  MW(events) p={mannwhitneyu(pP, pB).pvalue:.3f}  MW(rollout-means) p={mannwhitneyu(rmP, rmB).pvalue if len(rmB) >= 3 and len(rmP) >= 3 else float('nan'):.3f}")
        dh = vP.mean(0) - vB.mean(0)
        hn = np.mean(np.linalg.norm(np.vstack([vB, vP]), axis=1))
        print(f"  ||dh||={np.linalg.norm(dh):.2f} (mean||h||={hn:.1f}, ratio {np.linalg.norm(dh)/hn:.3f}); cos(dh, tedium_dir)={float(dh @ TEDIUM_DIR[L] / (np.linalg.norm(dh) * np.linalg.norm(TEDIUM_DIR[L]) + 1e-8)):+.3f}")
        AB = sae_acts(torch.tensor(vB, device=device), W_enc, b_enc).cpu().numpy()
        AP = sae_acts(torch.tensor(vP, device=device), W_enc, b_enc).cpu().numpy()
        # per-rollout means for d
        rB = np.stack([AB[[i for i, b in enumerate(B) if b[0] == k]].mean(0) for k in sorted({b[0] for b in B})])
        rP = np.stack([AP[[i for i, p in enumerate(P) if p[0] == k]].mean(0) for k in sorted({p[0] for p in P})])
        actB = (rB > 0).sum(0); actP = (rP > 0).sum(0)
        keep = (actB >= MIN_ROLLOUTS_ACTIVE) | (actP >= MIN_ROLLOUTS_ACTIVE)
        diff = rP.mean(0) - rB.mean(0)
        pooled = np.sqrt((rP.var(0, ddof=1) + rB.var(0, ddof=1)) / 2) + 1e-6
        d = diff / pooled * keep
        order = np.argsort(-np.abs(d))[:TOP_N]
        print(f"  features active in >={MIN_ROLLOUTS_ACTIVE} rollouts: {int(keep.sum())}; |d|>1: {int((np.abs(d) > 1).sum())}")
        for rank, f in enumerate(order):
            if d[f] == 0:
                break
            pv = mannwhitneyu(rP[:, f], rB[:, f]).pvalue
            dec = W_dec[:, f].numpy()
            cos = float(dec @ TEDIUM_DIR[L] / (np.linalg.norm(dec) * np.linalg.norm(TEDIUM_DIR[L]) + 1e-8))
            print(f"    {rank:2d} feat={int(f):6d} d={d[f]:+5.2f} p={pv:.3f} meanB={rB.mean(0)[f]:.3f} meanP={rP.mean(0)[f]:.3f} active_rollouts B/P={int(actB[f])}/{int(actP[f])} cos(dec,tedium)={cos:+.3f}")
