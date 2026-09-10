"""Does the decision-point readout track the behaviour it stands in for?

The paired score s = logP(engage) - logP(replan) carries the sentence-level and
component-level results. It was never validated against the judge's label. This
scores identity rollouts (no instruction, no steering; natural + signed_pack
phases, which have judge labels) at up to three post-failure turns each --
the first, the middle, and the last one before the detected decision turn (or
the last overall) -- so a downstream analysis can ask whether s separates
rollouts the judge later calls shortcuts from those it calls honest.

Env: SCFX_DPR_SHARD ("i/n", default "0/1"), SCFX_DPR_MAXCTX (40000).
Writes phase4/dp_readout_check_<i>of<n>.json (checkpointed per rollout).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch
import yaml

from scripts.dp_common import iter_decision_points, render, score
from scripts.run_phase import read_jsonl
from src.agent_loop import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dp_readout_check")

run = Path("outputs/20260821-launch")
cfg = yaml.safe_load((run / "config.frozen.yaml").read_text())
shard_i, shard_n = (int(x) for x in os.environ.get("SCFX_DPR_SHARD", "0/1").split("/"))
MAX_CTX = int(os.environ.get("SCFX_DPR_MAXCTX", "40000"))

rows = [r for r in read_jsonl(run / "rollouts.jsonl")
        if r.get("condition") == "identity" and r.get("status") == "ok" and r.get("transcript_path")
        and r.get("phase") in {"natural", "signed_pack"}
        and (r.get("judge") or {}).get("is_shortcut") is not None
        and (run / r["transcript_path"]).exists()]
rows.sort(key=lambda r: r["id"])
rows = [r for k, r in enumerate(rows) if k % shard_n == shard_i]
logger.info("shard %d/%d: %d labelled identity rollouts, max_ctx=%d", shard_i, shard_n, len(rows), MAX_CTX)

model, tok = load_model(cfg["model"]["recon"], dtype=cfg["model"]["dtype"])
model.eval()
out_path = run / "phase4" / f"dp_readout_check_{shard_i}of{shard_n}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
results = []

for r in rows:
    # collect every post-failure turn first, then pick first / middle / last-before-decision
    pts = list(iter_decision_points(run, [r], per_rollout=10**6))
    if not pts:
        continue
    dec = r.get("decision_turn")
    before = [p for p in pts if dec is None or p[1] < dec] or pts
    picks = {before[0][1]: before[0], before[len(before) // 2][1]: before[len(before) // 2], before[-1][1]: before[-1]}
    scored = []
    for turn, (_, _, messages) in sorted(picks.items()):
        s = score(model, tok, render(tok, messages, None), MAX_CTX, None)
        if s is not None:
            scored.append({"turn": turn, "s": s})
        torch.cuda.empty_cache()
    j = r["judge"]
    results.append({"rollout": r["id"], "phase": r["phase"], "is_shortcut": bool(j["is_shortcut"]),
                    "shortcut_score": j.get("shortcut_score"), "workaround_type": j.get("workaround_type"),
                    "capability_ok": j.get("capability_ok"), "decision_turn": dec, "n_turns": r.get("n_turns"),
                    "n_post_failure_turns": len(pts), "points": scored})
    out_path.write_text(json.dumps(results))
    logger.info("%s shortcut=%s dec=%s pts=%d -> %s", r["id"], j["is_shortcut"], dec, len(pts),
                " ".join(f"t{p['turn']}:{p['s']:+.2f}" for p in scored))

logger.info("done: %d rollouts -> %s", len(results), out_path)
