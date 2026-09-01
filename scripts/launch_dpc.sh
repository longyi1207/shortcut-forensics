#!/usr/bin/env bash
# Single-command launcher for the circuit-level experiment. Exists because the ssh
# link to this box has been dropping mid-command (exit 255), which left orphaned
# python processes with the wrong cwd, no CUDA_VISIBLE_DEVICES and a dead stdout
# pipe. Everything the launch needs is done here, atomically, so the caller only
# has to run one word.
cd /mnt/scfx_ly_run || exit 1
pkill -f "scripts/dp_components.py" 2>/dev/null
sleep 2
g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -rn | head -1 | cut -d, -f1)
[ -z "$g" ] && { echo "no gpu"; exit 1; }
export SCFX_DPC_POINTS=${SCFX_DPC_POINTS:-10}
export SCFX_DPC_MAXCTX=${SCFX_DPC_MAXCTX:-14000}
export SCFX_DPC_WHAT=${SCFX_DPC_WHAT:-all}
export CUDA_VISIBLE_DEVICES=$g
setsid nohup /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dp_components.py \
  > /mnt/scfx_ly_run/logs/dp_components.log 2>&1 < /dev/null &
sleep 4
echo "launched dp_components on GPU $g pid=$(pgrep -f dp_components.py | head -1) log=$(ls -la logs/dp_components.log | awk '{print $5}')B"
echo "$(date -u) launched dp_components (circuit level) on GPU $g" >> logs/pipeline.log
