#!/usr/bin/env bash
# VM-side hand-off: when the tug-of-war cell (pc_prompt_add_tedium19) reaches
# n>=30 ok rows, stop its workers and launch Stage 1 (dt_capture: dt_baseline x4,
# dt_prompt x4, n=20) on all 8 GPUs. Idempotent-ish: refuses to launch if
# dt_capture workers already exist.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  n=$(c prompt_channel pc_prompt_add_tedium19)
  if [ "$n" -ge 30 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage1: tug-of-war at n=$n -> stopping tw workers, launching Stage 1" | tee -a "$LOG"
for p in $(pgrep -f 'prompt_channe[l]'); do
  w=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in tw*) echo "stop $w pid=$p" | tee -a "$LOG"; kill "$p";; esac
done
sleep 30
if pgrep -f 'dt_captur[e]' >/dev/null; then echo "$(date -u) chain_stage1: dt_capture workers already running, not launching" | tee -a "$LOG"; exit 0; fi
i=0
for cond in dt_baseline dt_baseline dt_baseline dt_baseline dt_prompt dt_prompt dt_prompt dt_prompt; do
  (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$i SCFX_WORKER_ID=dt$i SCFX_DT_CONDITION=$cond SCFX_DT_N=20 \
     /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_capture.py > /mnt/scfx_ly_run/logs/dt_${cond}_dt$i.log 2>&1 < /dev/null &)
  echo "$(date -u) chain_stage1: launched $cond worker=dt$i GPU=$i" | tee -a "$LOG"
  i=$((i+1))
done
echo "$(date -u) chain_stage1: Stage 1 launched (8 workers)" | tee -a "$LOG"
