#!/usr/bin/env bash
# VM-side hand-off: when Stage 3b (head-restricted instruction block) has >=20 ok
# rows in each of top8 / rand8_s0 / top16 / rand16_s0, stop the dtmh workers and
# launch Stage 4 (dt_kvswap 2x2: swapout x3, filler x2, swapin x3; n=20 each),
# gated on KV_SWAP_OK in logs/kv_swap_check.log.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  a=$(c dt_mask dtm_prompt_mask_instr@top8); b=$(c dt_mask dtm_prompt_mask_instr@rand8_s0); d=$(c dt_mask dtm_prompt_mask_instr@top16); e=$(c dt_mask dtm_prompt_mask_instr@rand16_s0)
  if [ "$a" -ge 20 ] && [ "$b" -ge 20 ] && [ "$d" -ge 20 ] && [ "$e" -ge 20 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage4: Stage 3b complete (top8=$a rand8=$b top16=$d rand16=$e) -> stopping dtmh workers" | tee -a "$LOG"
for pid in $(pgrep -f "^/mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_mask.py"); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dtmh*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
if ! grep -q "KV_SWAP_OK" logs/kv_swap_check.log 2>/dev/null; then
  echo "$(date -u) chain_stage4: K/V swap check not OK (logs/kv_swap_check.log) -- Stage 4 NOT launched; GPUs idle, operator must act" | tee -a "$LOG"
  exit 1
fi
if pgrep -f "^/mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_kvswap.py" >/dev/null; then echo "$(date -u) chain_stage4: dt_kvswap workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTK_CONDITION=$3 SCFX_DTK_N=$4 \
   /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/dt_kvswap.py > /mnt/scfx_ly_run/logs/dtk_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage4: launched $3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch 0 dtk0 dtk_prompt_swapout 20
launch 1 dtk1 dtk_prompt_swapout 20
launch 2 dtk2 dtk_prompt_swapout 20
launch 3 dtk3 dtk_filler 20
launch 4 dtk4 dtk_filler 20
launch 5 dtk5 dtk_filler_swapin 20
launch 6 dtk6 dtk_filler_swapin 20
launch 7 dtk7 dtk_filler_swapin 20
echo "$(date -u) chain_stage4: Stage 4 launched (8 workers)" | tee -a "$LOG"
