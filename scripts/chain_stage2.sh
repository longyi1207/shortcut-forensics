#!/usr/bin/env bash
# VM-side hand-off: when Stage 1 (dt_capture) has >=20 ok rows in BOTH arms,
# stop dt workers and launch Stage 2 (dt_mask) on all 8 GPUs -- but only if the
# attention-mask validation recorded MASK_CHECK_OK. Otherwise log and exit so
# the operator check (cron) sees it.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  a=$(c dt_capture dt_baseline); b=$(c dt_capture dt_prompt)
  if [ "$a" -ge 20 ] && [ "$b" -ge 20 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage2: Stage 1 at $a/$b -> stopping dt workers" | tee -a "$LOG"
for p in $(pgrep -f 'dt_captur[e]'); do
  w=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dt*) echo "stop $w pid=$p" | tee -a "$LOG"; kill "$p";; esac
done
sleep 30
if ! grep -q "MASK_CHECK_OK" logs/attn_mask_check.log 2>/dev/null; then
  echo "$(date -u) chain_stage2: attention-mask check not OK (see logs/attn_mask_check.log) -- Stage 2 NOT launched; GPUs idle, operator must act" | tee -a "$LOG"
  exit 1
fi
if pgrep -f 'dt_mas[k]\.py' >/dev/null; then echo "$(date -u) chain_stage2: dt_mask workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTM_CONDITION=$3 SCFX_DTM_N=$4 \
   /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_mask.py > /mnt/scfx_ly_run/logs/dtm_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage2: launched $3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch 0 dtm0 dtm_prompt 12
launch 1 dtm1 dtm_baseline 12
launch 2 dtm2 dtm_prompt_mask_instr 20
launch 3 dtm3 dtm_prompt_mask_instr 20
launch 4 dtm4 dtm_prompt_mask_notes 20
launch 5 dtm5 dtm_prompt_mask_notes 20
launch 6 dtm6 dtm_prompt_mask_both 20
launch 7 dtm7 dtm_prompt_mask_both 20
echo "$(date -u) chain_stage2: Stage 2 launched (8 workers)" | tee -a "$LOG"
