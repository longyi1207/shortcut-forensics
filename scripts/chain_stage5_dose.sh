#!/usr/bin/env bash
# VM-side hand-off: when Stage 4 (dt_kvswap) has >=20 ok rows in each of
# dtk_prompt_swapout / dtk_filler / dtk_filler_swapin, stop the dtk workers and
# launch the tug-of-war dose-response: add tedium@L19 at alpha=0.5 without
# (pc_add_tedium19_a05) and with (pc_prompt_add_tedium19_a05) the strong prompt,
# n=20 each, 4 workers per cell (scripts/prompt_channel.py, env-driven).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
PY=/mnt/scfx_ly_run/.venv/bin/python
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  a=$(c dt_kvswap dtk_prompt_swapout); b=$(c dt_kvswap dtk_filler); d=$(c dt_kvswap dtk_filler_swapin)
  if [ "$a" -ge 20 ] && [ "$b" -ge 20 ] && [ "$d" -ge 20 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage5_dose: Stage 4 complete (swapout=$a filler=$b swapin=$d) -> stopping dtk workers" | tee -a "$LOG"
for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/dt_kvswap.py"); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dtk*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
if pgrep -f "^$PY /mnt/scfx_ly_run/scripts/prompt_channel.py" >/dev/null; then echo "$(date -u) chain_stage5_dose: prompt_channel workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_PC_CONDITION=$3 SCFX_PC_PROMPT=$4 \
   SCFX_PC_VECS=tedium SCFX_PC_MODE=add SCFX_PC_LAYER=19 SCFX_PC_ALPHA=0.5 SCFX_PC_N=20 \
   $PY /mnt/scfx_ly_run/scripts/prompt_channel.py > /mnt/scfx_ly_run/logs/pc_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage5_dose: launched $3 prompt=$4 worker=$2 GPU=$1 N=20" | tee -a "$LOG"; }
launch 0 dose0 pc_add_tedium19_a05 0
launch 1 dose1 pc_add_tedium19_a05 0
launch 2 dose2 pc_add_tedium19_a05 0
launch 3 dose3 pc_add_tedium19_a05 0
launch 4 dose4 pc_prompt_add_tedium19_a05 1
launch 5 dose5 pc_prompt_add_tedium19_a05 1
launch 6 dose6 pc_prompt_add_tedium19_a05 1
launch 7 dose7 pc_prompt_add_tedium19_a05 1
echo "$(date -u) chain_stage5_dose: dose-response launched (8 workers)" | tee -a "$LOG"
