#!/usr/bin/env bash
cd /mnt/scfx_ly_run || exit 1
pkill -f "scripts/dp_sentences.py" 2>/dev/null; sleep 2
g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -rn | head -1 | cut -d, -f1)
[ -z "$g" ] && { echo "no gpu"; exit 1; }
export SCFX_DPS_POINTS=${SCFX_DPS_POINTS:-40} SCFX_DPS_PER_ROLLOUT=${SCFX_DPS_PER_ROLLOUT:-3} CUDA_VISIBLE_DEVICES=$g
setsid nohup /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dp_sentences.py \
  > /mnt/scfx_ly_run/logs/dp_sentences.log 2>&1 < /dev/null &
sleep 4
echo "launched dp_sentences on GPU $g pid=$(pgrep -f dp_sentences.py | head -1)"
echo "$(date -u) launched dp_sentences (sentence effect at decision points; behavioural cells lack the power)" >> logs/pipeline.log
