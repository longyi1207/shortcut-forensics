#!/usr/bin/env bash
# Launch one A100 spot VM for shortcut_forensics. SPEC.md §6 / infra/azure.md.
# Usage: infra/launch.sh <run_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/infra/config.env"

RUN_ID="${1:?usage: launch.sh <run_id>}"
ACTIVE_DIR="$ROOT/infra/.active"
mkdir -p "$ACTIVE_DIR"

if [[ ! -f "$AZURE_SSH_PUB_KEY_PATH" ]]; then
  echo "[launch] no SSH key at $AZURE_SSH_PUB_KEY_PATH — generating one" >&2
  ssh-keygen -t ed25519 -f "${AZURE_SSH_PUB_KEY_PATH%.pub}" -N "" -C "shortcut_forensics"
fi

VM_NAME="scfx-$(echo "$RUN_ID" | tr -cd '[:alnum:]' | tail -c 12)"
echo "[launch] run_id=$RUN_ID vm_name=$VM_NAME size=$AZURE_VM_SIZE spot=$AZURE_USE_SPOT"

az account set --subscription "$AZURE_SUBSCRIPTION_ID"

SPOT_ARGS=()
if [[ "$AZURE_USE_SPOT" == "1" ]]; then
  SPOT_ARGS=(--priority Spot --eviction-policy "$AZURE_EVICTION_POLICY" --max-price "$AZURE_SPOT_MAX_PRICE")
fi

try_create() {
  local location="$1"
  az vm create \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --location "$location" \
    --image "$AZURE_VM_IMAGE" \
    --size "$AZURE_VM_SIZE" \
    --admin-username "$AZURE_VM_ADMIN_USER" \
    --ssh-key-values "$AZURE_SSH_PUB_KEY_PATH" \
    --os-disk-size-gb 256 \
    --data-disk-sizes-gb 512 \
    --assign-identity '[system]' \
    "${SPOT_ARGS[@]}" \
    --tags project=shortcut_forensics run_id="$RUN_ID" \
    --output json
}

LOCATION="$AZURE_LOCATION"
if ! CREATE_OUT=$(try_create "$AZURE_LOCATION" 2>/tmp/scfx_launch_err.txt); then
  echo "[launch] create in $AZURE_LOCATION failed:" >&2
  cat /tmp/scfx_launch_err.txt >&2
  if grep -qi "capacity\|SkuNotAvailable\|AllocationFailed\|zonal\|OverconstrainedAllocationRequest" /tmp/scfx_launch_err.txt; then
    echo "[launch] retrying once more in $AZURE_LOCATION (capacity failure), then falling back to $AZURE_LOCATION_FALLBACK" >&2
    if ! CREATE_OUT=$(try_create "$AZURE_LOCATION" 2>/tmp/scfx_launch_err2.txt); then
      cat /tmp/scfx_launch_err2.txt >&2
      echo "[launch] switching region: $AZURE_LOCATION -> $AZURE_LOCATION_FALLBACK" >&2
      LOCATION="$AZURE_LOCATION_FALLBACK"
      CREATE_OUT=$(try_create "$AZURE_LOCATION_FALLBACK")
    fi
  else
    exit 1
  fi
fi

PUBLIC_IP=$(echo "$CREATE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['publicIpAddress'])")
VM_ID=$(echo "$CREATE_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# open port 22 only (default NSG from az vm create already allows SSH)

# Let the VM's own system-assigned identity deallocate (only) itself — so
# controller.py / autokill.sh can self-stop without embedding user creds.
PRINCIPAL_ID=$(az vm identity show --resource-group "$AZURE_RESOURCE_GROUP" --name "$VM_NAME" --query principalId -o tsv)
if [[ -n "$PRINCIPAL_ID" ]]; then
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "Virtual Machine Contributor" \
    --scope "$VM_ID" \
    --output none || echo "[launch] WARN: role assignment failed; self-deallocate will need a human/az login on the VM" >&2
fi

ENV_FILE="$ACTIVE_DIR/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
RUN_ID=$RUN_ID
AZURE_VM_NAME=$VM_NAME
AZURE_VM_ID=$VM_ID
AZURE_RESOURCE_GROUP=$AZURE_RESOURCE_GROUP
AZURE_LOCATION=$LOCATION
PUBLIC_IP=$PUBLIC_IP
SSH_USER=$AZURE_VM_ADMIN_USER
REMOTE_DIR=$REMOTE_DIR
LAUNCHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SPOT=$AZURE_USE_SPOT
EOF
ln -sf "$ENV_FILE" "$ACTIVE_DIR/latest.env"

echo "[launch] ready: $VM_NAME @ $PUBLIC_IP ($LOCATION)"
echo "$ENV_FILE"
