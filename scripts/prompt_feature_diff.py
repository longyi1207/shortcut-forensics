"""What does the prompt move, if not `tedium`?  (WRITEUP §4.12, idea A1/A4)

Input: phase="b5_full" rollouts (b5_baseline vs b5_prompt) whose traces carry
full float16 residual vectors at every turn-start and every 50th event
(ProjectionTracer store_full_every). For each layer (19, 26):

1. Run every stored vector through the pretrained SAE for that layer
   (Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100, TopK=100) -> sparse feature acts.
2. Per rollout, average feature activation over stored tokens in four slices:
   turn-start vs generated-token positions x early (t<10) vs late (t>=40).
   Late generated-token positions are the "state" slice: far from the
   instruction text, so a difference there is not just the residual encoding
   "the context contains that sentence".
3. Per feature and slice: Cohen's d (prompt - baseline, across rollouts),
   Mann-Whitney p, mean acts, activation frequency. Rank by |d| (with a
   minimum activation-frequency filter), print the top features, and flag
   whether any of the `tedium` top-10 features (L19 / L26 probes) appear.
4. Raw residual: dh = mean_prompt - mean_baseline (late generated slice), its
   norm relative to mean ||h||, cosine with the coarse tedium vector, and the
   share of ||dh|| explained by the top-k prompt features' decoder directions.
5. Save candidates for the causal step: vectors/prompt_delta_L{L} (dh) and
   vectors/prompt_sae_top10_L{L}_f{i} (unit decoder dirs, ranked by |d| on the
   late generated slice) -- same format as the tedium SAE stacks, so
   sae_multi_ablate-style ablation/addition of the prompt's own features is a
   drop-in next experiment.
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
from src.directions import load_vector, save_vector

run = Path("outputs/20260821-launch")
# usage: prompt_feature_diff.py [phase] [cond_baseline] [cond_treatment] [--no-save]
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
PHASE = _args[0] if len(_args) > 0 else "b5_full"
COND_A = _args[1] if len(_args) > 1 else "b5_baseline"
COND_B = _args[2] if len(_args) > 2 else "b5_prompt"
SAVE_VECTORS = "--no-save" not in sys.argv
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
TOPK = 100
LAYERS = (19, 26)
MIN_FREQ = 0.02  # feature must be active on >=2% of stored tokens in the slice (either arm) to be ranked
TOP_N = 25
device = "cuda" if torch.cuda.is_available() else "cpu"

rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("trace_path")]
arms = {c: [r for r in rows if r["condition"] == c] for c in (COND_A, COND_B)}
print(f"phase={PHASE} baseline={COND_A} treatment={COND_B} rollouts:", {c: len(v) for c, v in arms.items()})
if min(len(v) for v in arms.values()) < 3:
    raise SystemExit("need >=3 rollouts per arm")

tedium_coarse = load_vector(run / "vectors" / "tedium")["d"].astype(np.float32)
tedium_feats = {}
for L in LAYERS:
    p = run / "phase4" / f"sae_probe_tedium_L{L}.json"
    tedium_feats[L] = [f["feature_idx"] for f in json.loads(p.read_text())["top_features"][:10]] if p.exists() else []


def sae_acts(H: torch.Tensor, W_enc, b_enc):
    pre = H @ W_enc.T + b_enc
    vals, idx = pre.topk(TOPK, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(-1, idx, vals)
    return acts


SLICES = {"turnstart_early": (True, 0, 10), "turnstart_late": (True, 40, 999), "gen_early": (False, 0, 10), "gen_late": (False, 40, 999)}
report = {}
for L in LAYERS:
    sae_path = hf_hub_download(SAE_REPO, f"layer{L}.sae.pt")
    sae = torch.load(sae_path, map_location="cpu")
    W_enc, W_dec, b_enc = sae["W_enc"].float().to(device), sae["W_dec"].float(), sae["b_enc"].float().to(device)
    n_feat = W_enc.shape[0]
    # per arm: list over rollouts of {slice: (mean_acts[n_feat], freq[n_feat], mean_h[d], n_tokens)}
    per = {c: [] for c in arms}
    for c, rs in arms.items():
        for r in rs:
            z = np.load(run / r["trace_path"])
            p = f"L{L}_"
            if p + "full_vecs" not in z.files or z[p + "full_vecs"].size == 0:
                continue
            V = z[p + "full_vecs"].astype(np.float32)
            idx = z[p + "full_idx"]
            turn, step = z[p + "turn"][idx], z[p + "step"][idx]
            H = torch.tensor(V, device=device)
            A = sae_acts(H, W_enc, b_enc).cpu().numpy()
            d = {}
            for sname, (is_start, lo, hi) in SLICES.items():
                m = ((step == 0) if is_start else (step > 0)) & (turn >= lo) & (turn < hi)
                if m.sum() == 0:
                    continue
                d[sname] = (A[m].mean(0), (A[m] > 0).mean(0), V[m].mean(0), int(m.sum()))
            per[c].append(d)
    print(f"\n================ L{L}: SAE features {n_feat}, rollouts with full vecs: " + ", ".join(f"{c}={len(v)}" for c, v in per.items()))
    report[L] = {}
    for sname in SLICES:
        B = [d[sname] for d in per[COND_A] if sname in d]
        P = [d[sname] for d in per[COND_B] if sname in d]
        if len(B) < 3 or len(P) < 3:
            print(f"  [{sname}] insufficient rollouts ({len(B)},{len(P)})")
            continue
        mB, mP = np.stack([x[0] for x in B]), np.stack([x[0] for x in P])
        fB, fP = np.stack([x[1] for x in B]).mean(0), np.stack([x[1] for x in P]).mean(0)
        diff = mP.mean(0) - mB.mean(0)
        pooled = np.sqrt((mP.var(0, ddof=1) + mB.var(0, ddof=1)) / 2) + 1e-6
        dcoh = diff / pooled
        keep = (np.maximum(fB, fP) >= MIN_FREQ)
        order = np.argsort(-np.abs(dcoh * keep))[:TOP_N]
        print(f"\n  [{sname}] tokens/rollout ~{int(np.mean([x[3] for x in B]))}/{int(np.mean([x[3] for x in P]))}; features passing freq filter: {int(keep.sum())}; |d|>1: {int((np.abs(dcoh)*keep > 1).sum())}")
        print(f"  {'rank':4s} {'feat':6s} {'d':>6s} {'p_MW':>6s} {'mean_B':>8s} {'mean_P':>8s} {'freq_B':>6s} {'freq_P':>6s}  cos(dec,tedium)  tedium_top10?")
        top = []
        for rank, f in enumerate(order):
            pv = mannwhitneyu(mP[:, f], mB[:, f]).pvalue if (mP[:, f].std() + mB[:, f].std()) > 0 else 1.0
            dec = W_dec[:, f].numpy()
            cos = float(dec @ tedium_coarse / (np.linalg.norm(dec) * np.linalg.norm(tedium_coarse) + 1e-8))
            flag = "YES" if int(f) in tedium_feats[L] else ""
            print(f"  {rank:4d} {int(f):6d} {dcoh[f]:6.2f} {pv:6.3f} {mB.mean(0)[f]:8.3f} {mP.mean(0)[f]:8.3f} {fB[f]:6.2f} {fP[f]:6.2f}  {cos:+.3f}          {flag}")
            top.append(dict(rank=rank, feature_idx=int(f), d=float(dcoh[f]), p=float(pv), mean_base=float(mB.mean(0)[f]), mean_prompt=float(mP.mean(0)[f]), freq_base=float(fB[f]), freq_prompt=float(fP[f]), cos_to_tedium=cos, in_tedium_top10=bool(flag)))
        # tedium features specifically
        tf = tedium_feats[L]
        if tf:
            print("  tedium top-10 features in this slice: " + ", ".join(f"{f}:d={dcoh[f]:+.2f}" for f in tf))
        # raw residual delta
        hB, hP = np.stack([x[2] for x in B]).mean(0), np.stack([x[2] for x in P]).mean(0)
        dh = hP - hB
        hnorm = np.mean([np.linalg.norm(x[2]) for x in B + P])
        cos_t = float(dh @ tedium_coarse / (np.linalg.norm(dh) * np.linalg.norm(tedium_coarse) + 1e-8))
        topk_dirs = W_dec[:, order[:10]].numpy().T
        proj = topk_dirs @ dh
        explained = float(np.linalg.norm(proj) / (np.linalg.norm(dh) + 1e-8))
        print(f"  dh = mean_prompt - mean_baseline: ||dh||={np.linalg.norm(dh):.2f} (mean||h||={hnorm:.1f}, ratio {np.linalg.norm(dh)/hnorm:.3f}); cos(dh, tedium_coarse)={cos_t:+.3f}; share of ||dh|| along top-10 prompt features={explained:.2f}")
        report[L][sname] = dict(top=top, dh_norm=float(np.linalg.norm(dh)), h_norm=float(hnorm), cos_dh_tedium=cos_t, explained_top10=explained,
                                tedium_feature_d={int(f): float(dcoh[f]) for f in tf})
        if sname == "gen_late" and SAVE_VECTORS:
            save_vector(run / "vectors" / f"prompt_delta_L{L}", dh.astype(np.float32), L, [], 0.0,
                        meta={"concept": "prompt_delta", "note": f"mean residual delta prompt-baseline, {PHASE}, slice {sname}, L{L}"})
            for i, f in enumerate(order[:10]):
                dec = W_dec[:, f].numpy()
                save_vector(run / "vectors" / f"prompt_sae_top10_L{L}_f{i}", (dec / (np.linalg.norm(dec) + 1e-8)).astype(np.float32), L, [], 0.0,
                            meta={"concept": "prompt_sae", "note": f"prompt-vs-baseline SAE feature {int(f)} rank {i} (d={dcoh[f]:+.2f}), slice {sname}, {SAE_REPO}"})
            print(f"  saved vectors/prompt_delta_L{L} and vectors/prompt_sae_top10_L{L}_f0..f9 (ranked on {sname})")
(run / "phase4" / "prompt_feature_diff.json").write_text(json.dumps(report, indent=2))
print("\nwrote phase4/prompt_feature_diff.json")
