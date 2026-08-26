#!/usr/bin/env bash
# Laptop-side: pull outputs/<run_id>/ from the box every INTERVAL_S while
# this loop runs. Meant to be launched once (in the background) after the
# box exists tomorrow and left running for the reservation window.
#
# Real limitation, not hidden: if the laptop sleeps, this loop pauses with
# it -- the blob backup (pull.sh --vm-side, wired into controller.py's own
# 15-min heartbeat, runs ON the box) is the safety net that does NOT depend
# on laptop uptime. This loop is the fast/convenient path for actually
# looking at results locally while working, not the only copy.
#
# Usage: infra/laptop_pull_loop.sh <run_id> [interval_seconds=600]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:?usage: laptop_pull_loop.sh <run_id> [interval_seconds]}"
INTERVAL_S="${2:-600}"

echo "[laptop_pull_loop] pulling outputs/$RUN_ID every ${INTERVAL_S}s (Ctrl-C to stop)"
while true; do
  bash "$ROOT/infra/pull.sh" "$RUN_ID" || true
  sleep "$INTERVAL_S"
done
