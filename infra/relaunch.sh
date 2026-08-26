#!/usr/bin/env bash
# VM-side: restart the controller in tmux after a Deallocate -> start cycle
# (spot eviction, autokill, or manual `az vm start`). Installed as an
# `@reboot` cron entry by bootstrap.sh, so this fires automatically on boot.
# Usage (on the VM): infra/relaunch.sh <run_id>
set -euo pipefail
RUN_ID="${1:?usage: relaunch.sh <run_id>}"
REMOTE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REMOTE_DIR"

# Give the network / NVIDIA driver a moment to settle after boot.
sleep 20

if tmux has-session -t scfx_ly 2>/dev/null; then
  echo "[relaunch] $(date -u -Iseconds) tmux session 'scfx_ly' already running, nothing to do"
  exit 0
fi

echo "[relaunch] $(date -u -Iseconds) starting controller for run_id=$RUN_ID"
mkdir -p "outputs/$RUN_ID/logs"
source .venv/bin/activate
source "$HOME/.scfx_env" 2>/dev/null || true  # HF_HOME etc, written by bootstrap.sh
tmux new-session -d -s scfx_ly \
  "cd $REMOTE_DIR && source .venv/bin/activate && source \$HOME/.scfx_env 2>/dev/null; python scripts/controller.py --run-id $RUN_ID 2>&1 | tee -a outputs/$RUN_ID/logs/controller.log"
echo "[relaunch] controller restarted"
