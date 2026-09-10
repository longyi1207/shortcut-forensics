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
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:$R/rejudge.jsonl" outputs/20260821-launch/rejudge.jsonl || true
OUT=writeup/RESULTS_$(date -u +%Y-%m-%d).md
{
  echo "# Prompt vs direction: results pulled $(date -u '+%Y-%m-%d %H:%MZ')"
  echo; echo "## 1. Behaviour under each concept's prompt line (vLLM, same-night baseline)"; echo '```'
  python3 scripts/sweep_analysis.py --min-id-prefix rvpd,rvpt,rvpb; echo '```'
  echo; echo "## 2. Per factor: prompting vs direction steering vs activation overlap"; echo '```'
  python3 scripts/factor_compare.py; echo '```'
  echo; echo "## 3. Decision-point grid per concept (readout proxy, tug-of-war, geometry)"; echo '```'
  python3 scripts/dp_concepts_analysis.py; echo '```'
  echo; echo "## 4. Geometry positive control: concept plants vs the instruction, every fitted layer, prompt end and decision token"; echo '```'
  python3 scripts/dp_plant_layers_analysis.py; echo '```'
  echo; echo "## 5. Does the decision-point readout track the judge label?"; echo '```'
  python3 scripts/dp_readout_analysis.py; echo '```'
  echo; echo "## 6. Figures"; echo '```'
  python3 scripts/fig_prompt_vs_steer.py 2>&1 | grep -v "UserWarning\|ax.legend"; echo '```'
  echo; echo "![behaviour](figs/prompt_vs_steer.png)"; echo; echo "![geometry](figs/geometry.png)"
} > "$OUT" 2>&1
echo "wrote $OUT"
