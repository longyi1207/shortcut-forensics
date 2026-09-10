#!/usr/bin/env bash
# Pull the complete experiment directory from the box to the laptop and verify it.
# Usage: infra/pull_all.sh            (from the repo root; safe to re-run, rsync is incremental)
# Pulls outputs/20260821-launch (rollouts, transcripts, diffs, judge_raw, activations, vectors, phase4,
# rejudge sidecar, incidents, ledgers) plus /mnt/scfx_logs, then checks that the local copy matches:
# rollouts.jsonl md5, row counts per phase, and that every ok row's transcript file exists locally.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source infra/.active/latest.env; set +a
KEY=$(grep -E '^SSH_KEY(_PATH)?=' infra/.active/latest.env | head -1 | cut -d= -f2-)
SSHE="ssh -i $KEY -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
R=/mnt/scfx_ly_run/outputs/20260821-launch
L=outputs/20260821-launch
mkdir -p "$L" outputs/vm_logs
echo "== rsync $R -> $L (incremental)"
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:$R/" "$L/"
echo "== rsync /mnt/scfx_logs -> outputs/vm_logs"
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:/mnt/scfx_logs/" outputs/vm_logs/
rsync -az -e "$SSHE" "$SSH_USER@$SSH_HOST:/mnt/scfx_ly_run/logs/" outputs/vm_logs/run_logs/
echo "== verify"
REMOTE_MD5=$($SSHE "$SSH_USER@$SSH_HOST" "md5sum $R/rollouts.jsonl | cut -d' ' -f1; wc -l < $R/rollouts.jsonl; ls $R/transcripts | wc -l")
LOCAL_MD5=$(md5 -q "$L/rollouts.jsonl" 2>/dev/null || md5sum "$L/rollouts.jsonl" | cut -d' ' -f1)
echo "remote md5/rows/transcripts: $(echo "$REMOTE_MD5" | tr '\n' ' ')"
echo "local  md5/rows/transcripts: $LOCAL_MD5 $(wc -l < "$L/rollouts.jsonl" | tr -d ' ') $(ls "$L/transcripts" | wc -l | tr -d ' ')"
[ "$(echo "$REMOTE_MD5" | head -1)" = "$LOCAL_MD5" ] && echo "rollouts.jsonl: MATCH" || { echo "rollouts.jsonl: MISMATCH (box still writing? re-run after the sweeps stop)"; }
python3 - "$L" <<'PY'
import json, sys, collections
from pathlib import Path
L = Path(sys.argv[1]); missing = 0; c = collections.Counter()
for line in open(L / "rollouts.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("status") != "ok":
        continue
    c[r.get("phase")] += 1
    tp = r.get("transcript_path")
    if tp and not (L / tp).exists():
        missing += 1
print("ok rows per phase:", dict(sorted(c.items())))
print("ok rows whose transcript is missing locally:", missing)
side = L / "rejudge.jsonl"
print("rejudge sidecar rows:", sum(1 for l in open(side) if l.strip()) if side.exists() else 0)
PY
du -sh "$L" | cut -f1 | sed 's/^/local size: /'
