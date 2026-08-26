#!/usr/bin/env bash
# Launch the Azure ML Compute Instance for the Startup GPU Cluster
# reservation (Italy North, NDm A100 v4, 2026-08-21..23). Distinct from
# launch.sh (raw az-vm-create path). Usage: infra/launch_aml.sh <run_id>
#
# Prerequisite: the quota support ticket (Quota type: Azure Machine
# Learning, region Italy North, SKU NDmA100v4) must already be approved —
# this will fail with a quota/SkuNotAvailable error otherwise. Workspace
# (scfx-ws-italynorth) is already created ahead of time; this script only
# creates the compute instance itself.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/infra/config.aml.env"

RUN_ID="${1:?usage: launch_aml.sh <run_id>}"
ACTIVE_DIR="$ROOT/infra/.active"
mkdir -p "$ACTIVE_DIR"

if [[ ! -f "$AML_SSH_PUB_KEY_PATH" ]]; then
  echo "[launch_aml] no SSH key at $AML_SSH_PUB_KEY_PATH — generate one first: ssh-keygen -t ed25519 -f ${AML_SSH_PUB_KEY_PATH%.pub}" >&2
  exit 1
fi

az account set --subscription "$AZURE_SUBSCRIPTION_ID"

echo "[launch_aml] creating compute instance $AML_COMPUTE_NAME ($AML_VM_SIZE) in workspace $AML_WORKSPACE ($AML_LOCATION)"

CREATE_OUT=$(az ml compute create \
  --name "$AML_COMPUTE_NAME" \
  --type ComputeInstance \
  --size "$AML_VM_SIZE" \
  --resource-group "$AML_RESOURCE_GROUP" \
  --workspace-name "$AML_WORKSPACE" \
  --ssh-public-access-enabled true \
  --ssh-key-value "$(cat "$AML_SSH_PUB_KEY_PATH")" \
  --enable-node-public-ip true \
  --identity-type SystemAssigned \
  --output json)

echo "$CREATE_OUT" > "$ACTIVE_DIR/${RUN_ID}.aml_create_raw.json"

# Grant the compute instance's own managed identity permission to stop
# itself (controller.py / autokill.sh call `az login --identity` then
# `az ml compute stop` from on-box). "AzureML Compute Operator" covers
# start/stop/restart on compute resources without broader Contributor
# access.
PRINCIPAL_ID=$(echo "$CREATE_OUT" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for path in ('identity.principal_id', 'identity.principalId'):
    cur = d
    try:
        for k in path.split('.'):
            cur = cur[k]
        if cur:
            print(cur); break
    except (KeyError, TypeError):
        continue
")
COMPUTE_RESOURCE_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AML_RESOURCE_GROUP}/providers/Microsoft.MachineLearningServices/workspaces/${AML_WORKSPACE}/computes/${AML_COMPUTE_NAME}"
STORAGE_RESOURCE_ID="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AML_RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${AML_STORAGE_ACCOUNT}"
if [[ -n "$PRINCIPAL_ID" ]]; then
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "AzureML Compute Operator" \
    --scope "$COMPUTE_RESOURCE_ID" \
    --output none || echo "[launch_aml] WARN: role assignment failed; self-stop will need a human/az login on the box" >&2
  # Lets pull.sh --vm-side back up outputs/ to blob storage periodically,
  # independent of laptop uptime and of whether this box has a real data disk.
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" \
    --scope "$STORAGE_RESOURCE_ID" \
    --output none || echo "[launch_aml] WARN: storage role assignment failed; blob backup will need a human/az login on the box" >&2
else
  echo "[launch_aml] WARN: could not extract managed identity principal_id from create output; self-stop will need a human/az login on the box" >&2
fi

echo "[launch_aml] querying connection details..."
SHOW_OUT=$(az ml compute show --name "$AML_COMPUTE_NAME" --resource-group "$AML_RESOURCE_GROUP" --workspace-name "$AML_WORKSPACE" --output json)
echo "$SHOW_OUT" > "$ACTIVE_DIR/${RUN_ID}.aml_show_raw.json"

# Field names vary between ML CLI v2's snake_case and the underlying ARM
# camelCase depending on az/extension version -- try both, and always keep
# the raw JSON above so a mismatch is a quick manual fix, not a mystery.
IFS=$'\t' read -r SSH_HOST SSH_PORT SSH_USER < <(python3 -c "
import json
d = json.load(open('$ACTIVE_DIR/${RUN_ID}.aml_show_raw.json'))
def get(*paths):
    for p in paths:
        cur = d
        try:
            for key in p.split('.'):
                cur = cur[key]
            if cur:
                return cur
        except (KeyError, TypeError):
            continue
    return ''
host = get('network_settings.public_ip_address', 'networkSettings.publicIpAddress', 'ssh_settings.ssh_host', 'sshSettings.sshHost', 'properties.sshSettings.sshHost')
port = get('ssh_settings.ssh_port', 'sshSettings.sshPort', 'properties.sshSettings.sshPort')
user = get('ssh_settings.admin_user_name', 'ssh_settings.admin_username', 'sshSettings.adminUserName') or '$AML_ADMIN_USER'
print(f'{host}\t{port}\t{user}')
")

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
  echo "[launch_aml] WARNING: could not auto-extract SSH host/port from az ml compute show output." >&2
  echo "[launch_aml] Read $ACTIVE_DIR/${RUN_ID}.aml_show_raw.json by hand, find the ssh host+port, and fill in $ACTIVE_DIR/${RUN_ID}.env manually." >&2
fi

ENV_FILE="$ACTIVE_DIR/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
RUN_ID=$RUN_ID
BACKEND=aml
AML_COMPUTE_NAME=$AML_COMPUTE_NAME
AML_RESOURCE_GROUP=$AML_RESOURCE_GROUP
AML_WORKSPACE=$AML_WORKSPACE
AML_LOCATION=$AML_LOCATION
AZURE_RESOURCE_GROUP=$AML_RESOURCE_GROUP
AZURE_LOCATION=$AML_LOCATION
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
SSH_USER=$SSH_USER
SSH_KEY_PATH=$AML_SSH_PRIVATE_KEY_PATH
REMOTE_DIR=$REMOTE_DIR
LAUNCHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RESERVATION_END_UTC=$AML_RESERVATION_END_UTC
EOF
ln -sf "$ENV_FILE" "$ACTIVE_DIR/latest.env"

echo "[launch_aml] ready: $AML_COMPUTE_NAME"
echo "[launch_aml] ssh -i $AML_SSH_PRIVATE_KEY_PATH -p ${SSH_PORT:-'<see raw json>'} ${SSH_USER}@${SSH_HOST:-'<see raw json>'}"
echo "$ENV_FILE"
