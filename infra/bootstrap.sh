#!/usr/bin/env bash
# Sync code + secrets to the VM, install deps, format/mount the data disk,
# install the relaunch cron, and start the controller in tmux.
# Usage: infra/bootstrap.sh <run_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AI_NOTES_ROOT="$(cd "$ROOT/../.." && pwd)"

RUN_ID="${1:?usage: bootstrap.sh <run_id>}"
ENV_FILE="$ROOT/infra/.active/${RUN_ID}.env"
[[ -f "$ENV_FILE" ]] || { echo "no $ENV_FILE — run launch.sh (or launch_aml.sh) first" >&2; exit 1; }
source "$ENV_FILE"

# BACKEND=aml is written by launch_aml.sh; unset/anything else means the
# original raw-VM path (launch.sh). Autokill hours: SPEC's default 8h for
# the raw-VM path, but the reservation's actual end date for the AML path
# (an 8h autokill would kill a ~48h-window reservation almost immediately).
if [[ "${BACKEND:-}" == "aml" ]]; then
  source "$ROOT/infra/config.aml.env"
  if [[ -n "${RESERVATION_END_UTC:-}" ]]; then
    AUTOKILL_HOURS=$(python3 -c "
from datetime import datetime, timezone
end = datetime.strptime('$RESERVATION_END_UTC', '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
hours = max(1, int((end - now).total_seconds() // 3600))
print(hours)
")
  else
    AUTOKILL_HOURS=44  # ~48h window minus safety margin, if RESERVATION_END_UTC somehow missing
  fi
  MAX_HOURS="$AUTOKILL_HOURS"
else
  source "$ROOT/infra/config.env"
fi

# Backend-agnostic connection: launch.sh (raw VM) writes PUBLIC_IP with
# default port 22; launch_aml.sh (Azure ML Compute Instance) writes
# SSH_HOST/SSH_PORT directly (non-standard port, e.g. 50000). Normalize
# here so bootstrap/pull/autokill don't need per-backend duplicates.
SSH_HOST="${SSH_HOST:-$PUBLIC_IP}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY_PATH:-${AZURE_SSH_PUB_KEY_PATH%.pub}}"
SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=15)
# scp's port flag is uppercase -P (lowercase -p means "preserve
# times/permissions" and takes no argument) -- reusing SSH_OPTS directly with
# scp silently mis-parses "-p 50000" as "-p" + a stray positional arg "50000",
# which scp then tries to stat as a local file. Separate array, not a typo fix
# in place, since ssh/rsync -e "ssh ..." both correctly want lowercase -p.
SCP_OPTS=(-i "$SSH_KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

log() { echo "[bootstrap@$SSH_HOST:$SSH_PORT] $*"; }

log "waiting for SSH ..."
for i in $(seq 1 60); do
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "echo ok" 2>/dev/null && break
  sleep 10
done

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "mkdir -p $REMOTE_DIR"

log "rsync code ..."
# --delete: without it, rsync only adds/updates -- it never removes files
# already present at the destination. On this box that meant an unrelated
# session's files (vendor/, eval_*.py, a different judge.py) silently
# survived into what looked like a fresh REMOTE_DIR and got imported
# instead of ours (2026-08-21 incident). --delete makes this an exact
# mirror of the local source, not a merge with whatever was already there.
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv' --exclude 'outputs' --exclude 'infra/.active/' \
  --exclude '__pycache__/' --exclude '*.pt' --exclude 'data/envs/agent-interp-envs/.git' \
  "$ROOT/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

# infra/.active/ is excluded above (it holds local-only launch bookkeeping,
# some of it laptop-relative), but controller.py's deallocate_self() and
# autokill.sh run ON this box and need this run's own identity (VM name /
# resource group, or AML compute/resource-group/workspace) to self-stop.
# Without this, self-stop silently fails to find anything -- copy just this
# one file over explicitly rather than syncing the whole (possibly stale,
# possibly laptop-path-referencing) directory.
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" "mkdir -p $REMOTE_DIR/infra/.active"
scp "${SCP_OPTS[@]}" "$ENV_FILE" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/infra/.active/${RUN_ID}.env"

log "sync minimal secrets (Azure OpenAI + HF token only, never the full .env) ..."
TMPENV=$(mktemp); chmod 600 "$TMPENV"
python3 - "$AI_NOTES_ROOT/.env" >"$TMPENV" <<'PYEOF'
import sys
path = sys.argv[1]
keep = {
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION", "OPENAI_PREFER_AZURE", "OPENAI_API_KEY",
    "HUGGING_FACE_TOKEN",
}
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k = line.split("=", 1)[0].strip()
        if k in keep:
            print(line)
PYEOF
scp "${SCP_OPTS[@]}" "$TMPENV" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/.env"
rm -f "$TMPENV"

log "remote setup: data disk, venv, deps, cron, tmux controller ..."
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" RUN_ID="$RUN_ID" REMOTE_DIR="$REMOTE_DIR" \
  VM_DATA_DISK_OUTPUTS_DIR="$VM_DATA_DISK_OUTPUTS_DIR" MAX_HOURS="$MAX_HOURS" HF_CACHE_DIR="${HF_CACHE_DIR:-}" bash -s <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"

# --- HF model cache: NOT the OS disk (~119GB, fills fast -- 3 experiments
# sharing this box each pulling multi-GB model weights into the default
# ~/.cache/huggingface hit 100% within minutes on 2026-08-21). Route to
# whatever large disk is actually mounted (found empirically per-image, not
# guessed) rather than the fragile /data auto-detect below, which can
# silently no-op if that disk is already mounted somewhere else (as it was
# here -- AML images pre-mount a large disk at /mnt).
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  sudo mkdir -p "$HF_CACHE_DIR"
  sudo chown -R "$USER" "$HF_CACHE_DIR"
  export HF_HOME="$HF_CACHE_DIR"
  echo "export HF_HOME=$HF_CACHE_DIR" >> "$HOME/.scfx_env"
fi

# --- data disk (best-effort; /data must survive a Deallocate/start cycle) ---
if ! mountpoint -q /data 2>/dev/null; then
  DISK_DEV=$(lsblk -ndo NAME,SIZE | awk '$2 ~ /5?[0-9][0-9]G|T/{print "/dev/"$1; exit}' | grep -v sda || true)
  if [[ -n "${DISK_DEV:-}" ]]; then
    sudo mkdir -p /data
    if ! sudo blkid "$DISK_DEV" >/dev/null 2>&1; then
      sudo mkfs.ext4 -F "$DISK_DEV"
    fi
    sudo mount "$DISK_DEV" /data || true
    grep -q "$DISK_DEV" /etc/fstab || echo "$DISK_DEV /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
  fi
fi
sudo mkdir -p "$VM_DATA_DISK_OUTPUTS_DIR"
# chown whatever VM_DATA_DISK_OUTPUTS_DIR actually resolves to -- NOT
# hardcoded /data. On AML (no real separate data disk, typically), this is
# a home-relative path (see config.aml.env), and the sudo mkdir above left
# it root-owned; without this the non-root controller/pull.sh process gets
# a silent "Permission denied" the first time it tries to write there.
sudo chown -R "$USER" "$VM_DATA_DISK_OUTPUTS_DIR" 2>/dev/null || true
sudo chown -R "$USER" /data 2>/dev/null || true

# --- python env ---
# `python3 -m venv` fails on this AML base image (ensurepip unavailable,
# and the matching python3.10-venv apt package 404s from the mirror -- a
# base-image issue, not something a retry fixes). AML compute instances
# ship pre-built conda envs instead; use one of those and shim a
# venv-compatible .venv/bin/{activate,python,pip} over it so every other
# script in this repo that does `source .venv/bin/activate` keeps working
# unmodified.
# Detection is unreliable here: venv creation gets far enough to leave an
# executable .venv/bin/python3 (passes -x checks) even when it's actually
# broken (ensurepip failed internally) -- confirmed on this box. Don't
# trust a plain-venv attempt; go straight to the known-working conda env.
rm -rf .venv
CONDA_ENV=/anaconda/envs/azureml_py310_sdkv2
if [[ -x "$CONDA_ENV/bin/python3" ]] && "$CONDA_ENV/bin/python3" -c "import ensurepip" 2>/dev/null; then
  mkdir -p .venv/bin
  ln -sf "$CONDA_ENV/bin/python3" .venv/bin/python
  ln -sf "$CONDA_ENV/bin/python3" .venv/bin/python3
  ln -sf "$CONDA_ENV/bin/pip" .venv/bin/pip
  cat > .venv/bin/activate <<ACTEOF
# minimal venv-compatible shim over a pre-built conda env -- just PATH, no
# conda internals (deactivate/env vars) since nothing here needs them.
export PATH="$(pwd)/.venv/bin:\$PATH"
ACTEOF
else
  python3 -m venv .venv
fi
source .venv/bin/activate
python --version
python -c "import sys; print('python OK:', sys.executable)"
pip install -q --upgrade pip
pip install -q -r requirements.txt

nvidia-smi -L || echo "WARNING: nvidia-smi not found — GPU image may not have loaded drivers yet"

mkdir -p "outputs/$RUN_ID/logs"

# --- relaunch cron + autokill safety net: best-effort, not fatal. This box
# has crontab locked down for this user (/etc/cron.allow denies it) -- both
# are redundancy (controller.py has its own autokill in its main loop; a
# reboot mid-reservation is not expected/planned for), not the primary
# mechanism, so losing them must not abort the rest of bootstrap (most
# importantly: actually starting the controller, below).
if command -v crontab >/dev/null 2>&1; then
  CRON_LINE="@reboot cd $REMOTE_DIR && bash infra/relaunch.sh $RUN_ID >> $REMOTE_DIR/outputs/$RUN_ID/logs/relaunch.log 2>&1"
  AUTOKILL_LINE="0 * * * * cd $REMOTE_DIR && bash infra/autokill.sh $RUN_ID $MAX_HOURS >> $REMOTE_DIR/outputs/$RUN_ID/logs/autokill.log 2>&1"
  if (crontab -l 2>/dev/null | grep -v "relaunch.sh $RUN_ID"; echo "$CRON_LINE") | crontab - 2>/dev/null && \
     (crontab -l 2>/dev/null | grep -v "autokill.sh $RUN_ID"; echo "$AUTOKILL_LINE") | crontab - 2>/dev/null; then
    echo "cron: relaunch + autokill installed"
  else
    echo "WARNING: crontab denied on this box (no cron.allow entry for this user) -- relying on controller.py's own autokill only, no reboot-survival cron. Not fatal, continuing." >&2
  fi
else
  echo "WARNING: crontab not available on this box -- relying on controller.py's own autokill only. Not fatal, continuing." >&2
fi

# --- start controller in tmux ---
tmux kill-session -t scfx_ly 2>/dev/null || true
tmux new-session -d -s scfx_ly "cd $REMOTE_DIR && source .venv/bin/activate && SCFX_AUTOKILL_HOURS=$MAX_HOURS python scripts/controller.py --run-id $RUN_ID 2>&1 | tee -a outputs/$RUN_ID/logs/controller.log"
echo "controller started in tmux session 'scfx_ly'"
REMOTE

log "bootstrap done. attach with: ssh -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} -t 'tmux attach -t scfx_ly'"
