#!/usr/bin/env bash
# VM-side hand-off: when Stage 3a (dt_heads) has dth_prompt>=12 ok rows, stop the
# dth workers, rank heads (scripts/dt_heads_analysis.py -> dt_heads_rank.json),
# and launch Stage 3b: instruction-span block restricted to the top-k reading
# heads vs size-matched random heads (dt_mask.py with SCFX_DTM_HEADS/HEADSET),
# n=20 each: top8, rand8_s0, top16, rand16_s0 (2 workers per set).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
RANK=outputs/20260821-launch/dt_heads_rank.json
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  n=$(c dt_heads dth_prompt)
  if [ "$n" -ge 12 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage3b: dt_heads at n=$n -> stopping dth workers" | tee -a "$LOG"
for pid in $(pgrep -f 'dt_head[s]\.py'); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dth*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
/mnt/scfx_ly_run/.venv/bin/python scripts/dt_heads_analysis.py > logs/dt_heads_analysis.log 2>&1
if [ ! -s "$RANK" ]; then
  echo "$(date -u) chain_stage3b: head ranking FAILED (no $RANK; see logs/dt_heads_analysis.log) -- Stage 3b NOT launched; GPUs idle, operator must act" | tee -a "$LOG"
  exit 1
fi
echo "$(date -u) chain_stage3b: ranking done: $(tail -1 logs/dt_heads_analysis.log)" | tee -a "$LOG"
if pgrep -f 'SCFX_DTM_HEADSE[T]' >/dev/null; then echo "$(date -u) chain_stage3b: head-ablation workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTM_CONDITION=dtm_prompt_mask_instr SCFX_DTM_HEADS=$RANK SCFX_DTM_HEADSET=$3 SCFX_DTM_N=$4 \
   /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_mask.py > /mnt/scfx_ly_run/logs/dtmh_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage3b: launched mask_instr@$3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch 0 dtmh0 top8 20
launch 1 dtmh1 top8 20
launch 2 dtmh2 rand8_s0 20
launch 3 dtmh3 rand8_s0 20
launch 4 dtmh4 top16 20
launch 5 dtmh5 top16 20
launch 6 dtmh6 rand16_s0 20
launch 7 dtmh7 rand16_s0 20
echo "$(date -u) chain_stage3b: Stage 3b launched (8 workers)" | tee -a "$LOG"
