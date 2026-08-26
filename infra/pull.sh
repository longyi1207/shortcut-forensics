#!/usr/bin/env bash
# Rsync outputs/<run_id>/ from the VM to the laptop, AND mirror it into the
# VM's own data disk (survives Deallocate; OS disk on spot eviction is also
# kept with Deallocate, but the data disk is the belt-and-suspenders copy).
# Usage: infra/pull.sh <run_id> [--vm-side]
#   (no flag)   run from the laptop: VM -> laptop outputs/
#   --vm-side   run ON the VM (called by controller.py): outputs/ -> /data mirror
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RUN_ID="${1:?usage: pull.sh <run_id> [--vm-side]}"
MODE="${2:-}"

if [[ "$MODE" == "--vm-side" ]]; then
  # Called by controller.py, already running on the VM/compute.
  IS_AML=0
  if [[ -f "$ROOT/infra/config.aml.env" ]] && [[ -f "$ROOT/infra/.active/${RUN_ID}.env" ]] && grep -q '^BACKEND=aml' "$ROOT/infra/.active/${RUN_ID}.env" 2>/dev/null; then
    source "$ROOT/infra/config.aml.env"
    IS_AML=1
  else
    source "$ROOT/infra/config.env"
  fi
  mkdir -p "$VM_DATA_DISK_OUTPUTS_DIR/$RUN_ID"
  rsync -a "$ROOT/outputs/$RUN_ID/" "$VM_DATA_DISK_OUTPUTS_DIR/$RUN_ID/" 2>/dev/null || true

  # Cloud-native backup: this box's local disk (mirrored above, or the OS
  # disk directly if there's no real separate data disk -- common for AML
  # compute instances) does NOT survive the compute being deleted (only a
  # plain stop). Laptop pulls don't survive the laptop sleeping either.
  # Blob storage does. *.pt activations excluded from this routine pass
  # (same reasoning as the laptop pull below) -- do one full backup
  # including them before intentionally spinning down.
  if [[ "$IS_AML" == "1" ]] && command -v az >/dev/null 2>&1; then
    az login --identity --output none 2>/dev/null || true
    az storage blob upload-batch \
      --account-name "$AML_STORAGE_ACCOUNT" \
      --destination "${AML_BACKUP_CONTAINER}/${RUN_ID}" \
      --source "$ROOT/outputs/$RUN_ID" \
      --pattern '*' --exclude-pattern '*.pt' \
      --auth-mode login --overwrite \
      --output none 2>/dev/null || echo "[pull --vm-side] blob backup failed (non-fatal, local mirror still done)" >&2
  fi
  exit 0
fi

ENV_FILE="$ROOT/infra/.active/${RUN_ID}.env"
[[ -f "$ENV_FILE" ]] || { echo "no $ENV_FILE — nothing to pull from" >&2; exit 1; }
source "$ENV_FILE"
if [[ "${BACKEND:-}" == "aml" ]]; then
  source "$ROOT/infra/config.aml.env"
else
  source "$ROOT/infra/config.env"
fi
SSH_HOST="${SSH_HOST:-$PUBLIC_IP}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY_PATH:-${AZURE_SSH_PUB_KEY_PATH%.pub}}"
SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

mkdir -p "$ROOT/outputs/$RUN_ID"
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  --exclude '*.pt' \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/outputs/${RUN_ID}/" "$ROOT/outputs/${RUN_ID}/" 2>&1 \
  || echo "[pull] rsync failed (box may be deallocated/stopped/evicted) — laptop copy may be stale"

echo "[pull] $(date -u -Iseconds) synced outputs/$RUN_ID (activations *.pt excluded — too big for routine pulls; fetch on demand)"
