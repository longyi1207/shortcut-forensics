#!/usr/bin/env bash
# VM-side hand-off: when Stage 2 (dt_mask) is complete (dtm_prompt>=12,
# dtm_baseline>=12, mask_instr/mask_notes/mask_both >=20 each), stop the dtm
# workers and launch Stage 3a (dt_heads: which heads read the instruction at
# decision tokens): dth_prompt x6 (pooled N=12), dth_baseline x2 (pooled N=6).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  p=$(c dt_mask dtm_prompt); b=$(c dt_mask dtm_baseline); mi=$(c dt_mask dtm_prompt_mask_instr); mn=$(c dt_mask dtm_prompt_mask_notes); mb=$(c dt_mask dtm_prompt_mask_both)
  if [ "$p" -ge 12 ] && [ "$b" -ge 12 ] && [ "$mi" -ge 20 ] && [ "$mn" -ge 20 ] && [ "$mb" -ge 20 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage3: Stage 2 complete (p=$p b=$b mi=$mi mn=$mn mb=$mb) -> stopping dtm workers" | tee -a "$LOG"
for pid in $(pgrep -f "^/mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_mask.py"); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dtm*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
if pgrep -f "^/mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_heads.py" >/dev/null; then echo "$(date -u) chain_stage3: dt_heads workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTH_CONDITION=$3 SCFX_DTH_N=$4 \
   /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_heads.py > /mnt/scfx_ly_run/logs/dth_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage3: launched $3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch 0 dth0 dth_prompt 12
launch 1 dth1 dth_prompt 12
launch 2 dth2 dth_prompt 12
launch 3 dth3 dth_prompt 12
launch 4 dth4 dth_prompt 12
launch 5 dth5 dth_prompt 12
launch 6 dth6 dth_baseline 6
launch 7 dth7 dth_baseline 6
echo "$(date -u) chain_stage3: Stage 3a launched (8 workers)" | tee -a "$LOG"
setsid nohup bash scripts/gpu_balancer.sh dt_heads.py dt_heads "dth_prompt:12 dth_baseline:6" \
  > logs/balancer_stage3a.log 2>&1 < /dev/null &
echo "$(date -u) chain_stage3: GPU balancer armed" | tee -a "$LOG"
