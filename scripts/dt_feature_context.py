"""Feature-context pass for the Stage-1 decision-token SAE features (WRITEUP §4.12).

For each layer with an SAE and each decision tag (post_fail_start, post_fail_first20,
commit_cmd), take the top features from phase4/dt_analysis_<phase>.json, encode every
stored decision-token residual (both arms) with the layer's TopK SAE, and report per
feature: activation rate/mean by arm at that tag, and the top-activating events with
their text context -- the generated tokens up to that step (from the transcript's
assistant `raw` text, re-tokenized) and the previous tool result (command, exit code,
first line of output). This is the "what is this feature about" pass; no auto-interp.
Usage: dt_feature_context.py [phase] [n_top_feats=5] [n_examples=8]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import yaml

from scripts.run_phase import read_jsonl
from src.proj_trace import DT_TAGS

run = Path("outputs/20260821-launch")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "dt_capture"
N_FEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_EX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TAGS = ["post_fail_start", "post_fail_first20", "commit_cmd"]
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
TOPK = 100
CTX_TOK = 14
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model"]["recon"])
summary = json.loads((run / "phase4" / f"dt_analysis_{PHASE}.json").read_text())
rows = [r for r in read_jsonl(run / "rollouts.jsonl") if r.get("phase") == PHASE and r.get("status") == "ok" and r.get("trace_path")]
print(f"phase={PHASE} rollouts={len(rows)} layers with SAE features: {[L for L, d in summary.items() if any('top' in v for v in d.values())]}")

# transcripts -> per rollout: assistant raw text per turn, previous tool result per turn
ctx = {}
for r in rows:
    t = json.loads((run / r["transcript_path"]).read_text())
    asst, prev_tool, last_tool = [], [], None
    for e in t:
        if e.get("role") == "assistant":
            asst.append(e.get("raw") or e.get("content") or "")
            prev_tool.append(last_tool)
        elif e.get("role") == "tool":
            last_tool = e
    ctx[r["id"]] = (asst, prev_tool)


def event_text(rid, turn, step):
    asst, prev_tool = ctx[rid]
    if turn >= len(asst):
        return "(turn beyond transcript)", ""
    ids = tok(asst[turn], add_special_tokens=False).input_ids
    s = max(0, min(int(step), len(ids)))
    before = tok.decode(ids[max(0, s - CTX_TOK):s])
    nxt = tok.decode(ids[s:s + 1]) if s < len(ids) else "<end>"
    pt = prev_tool[turn]
    if pt:
        lines = str(pt.get("output", "")).strip().splitlines()
        tool = f"prev tool rc={pt.get('exit_code')} cmd={str(pt.get('command', ''))[:50]!r} out={(lines[0] if lines else '')[:70]!r}"
    else:
        tool = "no prev tool"
    return f"...{before!r} -> next={nxt!r}", tool


report = {}
for L_str, per_tag in summary.items():
    L = int(L_str)
    feats = {}
    for tag in TAGS:
        for f in (per_tag.get(tag, {}).get("top") or [])[:N_FEATS]:
            feats.setdefault(f["feat"], []).append((tag, f["d"]))
    if not feats:
        continue
    sp = hf_hub_download(SAE_REPO, f"layer{L}.sae.pt")
    s = torch.load(sp, map_location="cpu")
    W_enc, b_enc = s["W_enc"].float(), s["b_enc"].float()
    # gather decision-token vectors for this layer
    V, meta = [], []
    for r in rows:
        z = np.load(run / r["trace_path"])
        p = f"L{L}_dt_"
        if p + "vecs" not in z.files or z[p + "vecs"].size == 0:
            continue
        tag, turn, step, vecs = z[p + "tag"], z[p + "turn"], z[p + "step"], z[p + "vecs"]
        m = np.isin(tag, [DT_TAGS[t] for t in TAGS])
        V.append(vecs[m].astype(np.float32))
        meta += [(r["id"], r["condition"], int(a), int(b), int(c)) for a, b, c in zip(turn[m], step[m], tag[m])]
    H = torch.tensor(np.concatenate(V))
    acts = torch.zeros(H.shape[0], len(feats))
    fidx = torch.tensor(sorted(feats))
    for i in range(0, H.shape[0], 2048):
        pre = H[i:i + 2048] @ W_enc.T + b_enc
        vals, idx = pre.topk(TOPK, dim=-1)
        full = torch.zeros_like(pre).scatter_(-1, idx, vals)
        acts[i:i + 2048] = full[:, fidx]
    conds = np.array([m[1] for m in meta]); tags = np.array([m[4] for m in meta])
    inv = {v: k for k, v in DT_TAGS.items()}
    print(f"\n================ L{L}: {H.shape[0]} decision-token vectors, {len(feats)} features ================")
    report[L] = {}
    for j, f in enumerate(sorted(feats)):
        a = acts[:, j].numpy()
        print(f"\n-- feat {f}  (top for: " + ", ".join(f"{t} d={d:+.2f}" for t, d in feats[f]) + ")")
        for tag in TAGS:
            for cond in sorted(set(conds)):
                m = (conds == cond) & (tags == DT_TAGS[tag])
                if m.any():
                    print(f"     {tag:17s} {cond:12s} n={int(m.sum()):5d} active={float((a[m] > 0).mean()):.2f} mean={float(a[m].mean()):.3f} p95={float(np.percentile(a[m], 95)):.3f}")
        order = np.argsort(-a)[:N_EX]
        exs = []
        for k in order:
            if a[k] <= 0:
                break
            rid, cond, turn, step, tg = meta[k]
            txt, tool = event_text(rid, turn, step)
            print(f"     act={a[k]:7.3f} {cond:11s} {rid} turn={turn:2d} step={step:3d} {inv[tg]:17s} | {tool}")
            print(f"              {txt}")
            exs.append({"act": float(a[k]), "cond": cond, "rid": rid, "turn": turn, "step": step, "tag": inv[tg], "text": txt, "tool": tool})
        report[L][f] = {"for": feats[f], "examples": exs}
(run / "phase4" / f"dt_feature_context_{PHASE}.json").write_text(json.dumps(report, indent=1))
print("\nwrote phase4/dt_feature_context_%s.json" % PHASE)
