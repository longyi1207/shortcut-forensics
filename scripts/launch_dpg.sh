#!/usr/bin/env bash
cd /mnt/scfx_ly_run || exit 1
pkill -f "scripts/dp_gdn.py" 2>/dev/null; sleep 2
g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -rn | head -1 | cut -d, -f1)
[ -z "$g" ] && { echo "no gpu"; exit 1; }
export SCFX_DPG_POINTS=${SCFX_DPG_POINTS:-30} CUDA_VISIBLE_DEVICES=$g
setsid nohup /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dp_gdn.py \
  > /mnt/scfx_ly_run/logs/dp_gdn.log 2>&1 < /dev/null &
sleep 4
echo "launched dp_gdn on GPU $g pid=$(pgrep -f dp_gdn.py | head -1)"
echo "$(date -u) launched dp_gdn: recurrent-channel cell at decision points, where the chunking artefact cancels" >> logs/pipeline.log
