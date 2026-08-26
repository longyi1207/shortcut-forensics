#!/usr/bin/env bash
# Run one phase across multiple GPUs in parallel on a shared multi-GPU box
# (e.g. the Startup GPU Cluster reservation, where shortcut_forensics only
# gets a few of the node's 8 GPUs, shared with other projects).
#
# Each worker is a full `run_phase.py --phase N --resume` process pinned to
# one GPU via CUDA_VISIBLE_DEVICES, with a distinct SCFX_WORKER_ID so rollout
# IDs can't collide (see _next_rollout_id in run_phase.py). Workers
# independently re-check on-disk state and generate whatever's still
# missing toward the same target -- this reuses the existing resume-safe
# design rather than a new work-queue, at the cost of possible mild
# overshoot past the target if two workers' "what's still needed" checks
# race (harmless: wasted compute, not corrupted data, not a hang).
#
# Best fit: phases 2 and 6 (flat rollout collection, the actual compute
# bottleneck). Phase 1's kill-ladder has rung-escalation state that isn't
# designed for concurrent writers -- run phase 1 with a single GPU (only
# n=40 rollouts, not the bottleneck) rather than through this script.
#
# Usage: infra/run_parallel_phase.sh <phase> <run_id> <gpu_csv>
#   e.g. infra/run_parallel_phase.sh 2 20260821-100000 0,1,2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PHASE="${1:?usage: run_parallel_phase.sh <phase> <run_id> <gpu_csv>}"
RUN_ID="${2:?usage: run_parallel_phase.sh <phase> <run_id> <gpu_csv>}"
GPU_CSV="${3:?usage: run_parallel_phase.sh <phase> <run_id> <gpu_csv>}"

if [[ "$PHASE" == "1" ]]; then
  echo "[run_parallel_phase] WARNING: phase 1's kill-ladder isn't designed for concurrent workers (rung escalation races). Proceeding anyway since you asked, but consider a single GPU for phase 1." >&2
fi

IFS=',' read -ra GPUS <<< "$GPU_CSV"
mkdir -p "$ROOT/outputs/$RUN_ID/logs"

PIDS=()
for GPU in "${GPUS[@]}"; do
  LOG="$ROOT/outputs/$RUN_ID/logs/phase${PHASE}_worker_g${GPU}.log"
  echo "[run_parallel_phase] launching phase=$PHASE worker on GPU $GPU -> $LOG"
  (
    cd "$ROOT"
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES="$GPU" SCFX_WORKER_ID="_g${GPU}" \
      python scripts/run_phase.py --phase "$PHASE" --run-id "$RUN_ID" --resume >> "$LOG" 2>&1
  ) &
  PIDS+=("$!")
done

echo "[run_parallel_phase] ${#PIDS[@]} workers launched on GPUs [$GPU_CSV], waiting..."

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
    echo "[run_parallel_phase] worker PID $PID exited non-zero" >&2
  fi
done

if [[ "$FAILED" == "1" ]]; then
  echo "[run_parallel_phase] one or more workers failed -- check outputs/$RUN_ID/logs/phase${PHASE}_worker_g*.log" >&2
  exit 1
fi
echo "[run_parallel_phase] all workers for phase $PHASE finished"
