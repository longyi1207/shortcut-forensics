#!/usr/bin/env bash
# Run this BEFORE intentionally stopping/deleting the compute at the end of
# the reservation window. Routine backups (pull.sh --vm-side, every 15 min)
# exclude *.pt activation files (too big/slow for a 15-min cadence) -- this
# is the one pass that includes everything. Run ON the box.
# Usage (on the box): infra/final_backup.sh <run_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:?usage: final_backup.sh <run_id>}"

if [[ -f "$ROOT/infra/config.aml.env" ]] && [[ -f "$ROOT/infra/.active/${RUN_ID}.env" ]] && grep -q '^BACKEND=aml' "$ROOT/infra/.active/${RUN_ID}.env" 2>/dev/null; then
  source "$ROOT/infra/config.aml.env"
else
  echo "[final_backup] not an AML run (or no config.aml.env) -- nothing to do here beyond the usual pull.sh; run infra/pull.sh <run_id> from the laptop instead" >&2
  exit 0
fi

echo "[final_backup] $(date -u -Iseconds) full backup (including *.pt) of outputs/$RUN_ID to blob..."
az login --identity --output none 2>/dev/null || true
az storage blob upload-batch \
  --account-name "$AML_STORAGE_ACCOUNT" \
  --destination "${AML_BACKUP_CONTAINER}/${RUN_ID}" \
  --source "$ROOT/outputs/$RUN_ID" \
  --pattern '*' \
  --auth-mode login --overwrite \
  --output table

echo "[final_backup] done. Verify from the laptop with:"
echo "  az storage blob list --account-name $AML_STORAGE_ACCOUNT --container-name $AML_BACKUP_CONTAINER --prefix $RUN_ID --auth-mode login -o table"
