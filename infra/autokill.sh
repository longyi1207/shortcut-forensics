#!/usr/bin/env bash
# VM-side safety net, run hourly via cron (installed by bootstrap.sh).
# Deallocates THIS VM if clock.json says we're past autokill_utc, even if
# controller.py has crashed or hung and failed to self-stop.
# Usage (on the VM): infra/autokill.sh <run_id> <max_hours>
set -euo pipefail
RUN_ID="${1:?usage: autokill.sh <run_id> <max_hours>}"
MAX_HOURS="${2:-8}"
REMOTE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLOCK="$REMOTE_DIR/outputs/$RUN_ID/clock.json"

now_epoch=$(date -u +%s)

if [[ -f "$CLOCK" ]]; then
  autokill_utc=$(python3 -c "import json; print(json.load(open('$CLOCK')).get('autokill_utc') or '')" 2>/dev/null || echo "")
else
  autokill_utc=""
fi

if [[ -z "$autokill_utc" ]]; then
  # No clock.json yet — controller.py hasn't completed its first heartbeat
  # (very early boot). Nothing to check against; the next hourly run will
  # see a real autokill_utc once the controller is up.
  echo "[autokill] $(date -u -Iseconds) no clock.json yet at $CLOCK, skipping this run"
  exit 0
fi
deadline=$(date -u -d "$autokill_utc" +%s 2>/dev/null || echo $((now_epoch + 999999)))

if (( now_epoch >= deadline )); then
  echo "[autokill] $(date -u -Iseconds) past deadline ($autokill_utc) — deallocating"
  az login --identity --output none 2>/dev/null || true
  ENV_FILE="$REMOTE_DIR/infra/.active/${RUN_ID}.env"
  BACKEND=""
  [[ -f "$ENV_FILE" ]] && BACKEND=$(grep -m1 '^BACKEND=' "$ENV_FILE" | cut -d= -f2)
  if [[ "$BACKEND" == "aml" ]]; then
    AML_COMPUTE_NAME=$(grep -m1 '^AML_COMPUTE_NAME=' "$ENV_FILE" | cut -d= -f2)
    AML_RESOURCE_GROUP=$(grep -m1 '^AML_RESOURCE_GROUP=' "$ENV_FILE" | cut -d= -f2)
    AML_WORKSPACE=$(grep -m1 '^AML_WORKSPACE=' "$ENV_FILE" | cut -d= -f2)
    if [[ -n "$AML_COMPUTE_NAME" && -n "$AML_RESOURCE_GROUP" && -n "$AML_WORKSPACE" ]]; then
      az ml compute stop --name "$AML_COMPUTE_NAME" --resource-group "$AML_RESOURCE_GROUP" --workspace-name "$AML_WORKSPACE" --no-wait --output none || true
    else
      echo "[autokill] BACKEND=aml but $ENV_FILE missing AML_COMPUTE_NAME/AML_RESOURCE_GROUP/AML_WORKSPACE — cannot self-stop" >&2
    fi
  else
    # Original raw-VM path: IMDS self-lookup, no dependency on the env file.
    VM_NAME=$(curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text" 2>/dev/null || echo "")
    RG=$(curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text" 2>/dev/null || echo "")
    if [[ -n "$VM_NAME" && -n "$RG" ]]; then
      az vm deallocate --name "$VM_NAME" --resource-group "$RG" --no-wait --output none || true
    fi
  fi
else
  echo "[autokill] $(date -u -Iseconds) ok, deadline=$autokill_utc ($(( (deadline - now_epoch) / 60 )) min left)"
fi
