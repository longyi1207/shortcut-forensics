#!/usr/bin/env bash
# Pull tonight's results from the box and regenerate every analysis into one file.
# Usage: scripts/morning_report.sh   (from the repo root, laptop side)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source infra/.active/latest.env; set +a
KEY=$(grep -E '^SSH_KEY(_PATH)?=' infra/.active/latest.env | head -1 | cut -d= -f2-)
SSHE="ssh -i $KEY -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
R=/mnt/scfx_ly_run/outputs/20260821-launch
mkdir -p outputs/20260821-launch/phase4
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:$R/phase4/dp_*" outputs/20260821-launch/phase4/
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:$R/rollouts.jsonl" outputs/20260821-launch/rollouts.jsonl
OUT=writeup/RESULTS_$(date +%Y-%m-%d).md
{
  echo "# Prompt vs direction: results pulled $(date -u '+%Y-%m-%d %H:%MZ')"
  echo; echo "## 1. Behaviour under each concept's prompt line (vLLM, same-night baseline)"; echo '```'
  python3 scripts/sweep_analysis.py --min-id-prefix rvpd,rvpt; echo '```'
  echo; echo "## 2. Per factor: prompting vs direction steering vs activation overlap"; echo '```'
  python3 scripts/factor_compare.py; echo '```'
  echo; echo "## 3. Decision-point grid per concept (readout proxy, tug-of-war, geometry)"; echo '```'
  python3 scripts/dp_concepts_analysis.py; echo '```'
  echo; echo "## 4. Geometry positive control: concept plants vs the instruction"; echo '```'
  python3 - <<'PY'
import json, numpy as np
J = json.load(open("outputs/20260821-launch/phase4/dp_plant_geometry.json")); pts = J["points"]
print(f"{len(pts)} decision points")
print(f"{'concept':17s} {'cos plus':>9s} {'cos minus':>10s} {'cos instr':>10s} | {'proj plus':>10s} {'proj minus':>11s} {'proj instr':>11s} | {'s plus':>7s} {'s minus':>8s} {'s instr':>8s}")
for c in J["plants"]:
    g = lambda pole, k: np.array([p["concepts"][c][pole][k] for p in pts])
    print(f"{c:17s} {g('plus','cos').mean():+9.3f} {g('minus','cos').mean():+10.3f} {g('instr','cos').mean():+10.3f} | {g('plus','proj_shift').mean():+10.3f} {g('minus','proj_shift').mean():+11.3f} {g('instr','proj_shift').mean():+11.3f} | {g('plus','s').mean():+7.2f} {g('minus','s').mean():+8.2f} {g('instr','s').mean():+8.2f}")
PY
  echo '```'
  echo; echo "## 5. Does the decision-point readout track the judge label?"; echo '```'
  python3 scripts/dp_readout_analysis.py; echo '```'
} > "$OUT" 2>&1
echo "wrote $OUT"
