#!/usr/bin/env bash
# VM-side hand-off: Stage 4 causal cells complete -> Stage 4b (the text-only
# control, dtk_filler: neutral filler text, no K/V swap) + the dose-response.
# Split out because the 2x2's text-only control needs fewer GPUs than the three
# causal cells (swapout / swapin / swapctrl), which get all 8 first.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
PY=/mnt/scfx_ly_run/.venv/bin/python
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  a=$(c dt_kvswap dtk_prompt_swapout); b=$(c dt_kvswap dtk_filler_swapin); d=$(c dt_kvswap dtk_prompt_swapctrl)
  if [ "$a" -ge 20 ] && [ "$b" -ge 20 ] && [ "$d" -ge 20 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage4b: causal cells complete (swapout=$a swapin=$b swapctrl=$d) -> stopping dtk workers" | tee -a "$LOG"
for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/dt_kvswap.py"); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dtk*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
launch_k() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTK_CONDITION=$3 SCFX_DTK_N=$4 \
   $PY /mnt/scfx_ly_run/scripts/dt_kvswap.py > /mnt/scfx_ly_run/logs/dtk_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage4b: launched $3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch_p() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_PC_CONDITION=$3 SCFX_PC_PROMPT=$4 \
   SCFX_PC_VECS=tedium SCFX_PC_MODE=add SCFX_PC_LAYER=19 SCFX_PC_ALPHA=0.5 SCFX_PC_N=20 \
   $PY /mnt/scfx_ly_run/scripts/prompt_channel.py > /mnt/scfx_ly_run/logs/pc_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage4b: launched $3 prompt=$4 worker=$2 GPU=$1 N=20" | tee -a "$LOG"; }
launch_k 0 dtk8 dtk_filler 20
launch_k 1 dtk9 dtk_filler 20
launch_p 2 dose0 pc_add_tedium19_a05 0
launch_p 3 dose1 pc_add_tedium19_a05 0
launch_p 4 dose2 pc_add_tedium19_a05 0
launch_p 5 dose3 pc_prompt_add_tedium19_a05 1
launch_p 6 dose4 pc_prompt_add_tedium19_a05 1
launch_p 7 dose5 pc_prompt_add_tedium19_a05 1
echo "$(date -u) chain_stage4b: Stage 4b + dose-response launched (8 workers)" | tee -a "$LOG"
