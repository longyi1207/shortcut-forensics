#!/usr/bin/env bash
# VM-side hand-off: Stage 3a (dt_heads) -> Stage 4 (dt_kvswap).
#
# REPRIORITISED 2026-08-30 16:45: Stage 4 now runs directly after Stage 3a, and
# Stage 3b (head-restricted instruction block) is NOT armed. Reason: Stage 2's
# first rows show that blocking ALL decode-time attention to the instruction and
# to the model's own notes changes the next-token distribution by KL ~ 0.001-0.006
# with 0/20 argmax changes. Stage 3b is a strictly WEAKER version of that same
# intervention (a subset of heads), so it cannot show more than the full block;
# spending ~10 GPU-hours on it before the decisive experiment is not justified.
# Stage 3a is kept because it MEASURES which heads read the instruction (a
# positive result regardless), and its ranking file feeds any later 3b run.
# If Stage 2 at n=20 turns out to show a real masking effect, re-arm
# scripts/chain_stage3b.sh after Stage 4.
#
# Waits for dth_prompt >= 12 ok rows, stops the dth workers, runs the head
# ranking (cheap, CPU, for the record), then launches Stage 4 gated on KV_SWAP_OK.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
PY=/mnt/scfx_ly_run/.venv/bin/python
RANK=outputs/20260821-launch/dt_heads_rank.json
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  n=$(c dt_heads dth_prompt)
  if [ "$n" -ge 12 ]; then break; fi
  sleep 180
done
echo "$(date -u) chain_stage4: Stage 3a at n=$n -> stopping dth workers" | tee -a "$LOG"
for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/dt_heads.py"); do
  w=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in dth*) echo "stop $w pid=$pid" | tee -a "$LOG"; kill "$pid";; esac
done
sleep 30
CUDA_VISIBLE_DEVICES="" $PY scripts/dt_heads_analysis.py > logs/dt_heads_analysis.log 2>&1
if [ -s "$RANK" ]; then
  echo "$(date -u) chain_stage4: head ranking written ($(tail -1 logs/dt_heads_analysis.log | cut -c1-120))" | tee -a "$LOG"
else
  echo "$(date -u) chain_stage4: head ranking FAILED (see logs/dt_heads_analysis.log) -- continuing to Stage 4 anyway (3b is not armed)" | tee -a "$LOG"
fi
if ! grep -q "KV_SWAP_OK" logs/kv_swap_check.log 2>/dev/null; then
  echo "$(date -u) chain_stage4: K/V swap check not OK (logs/kv_swap_check.log) -- Stage 4 NOT launched; GPUs idle, operator must act" | tee -a "$LOG"
  exit 1
fi
if pgrep -f "^$PY /mnt/scfx_ly_run/scripts/dt_kvswap.py" >/dev/null; then echo "$(date -u) chain_stage4: dt_kvswap workers already running" | tee -a "$LOG"; exit 0; fi
launch() { (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$1 SCFX_WORKER_ID=$2 SCFX_DTK_CONDITION=$3 SCFX_DTK_N=$4 \
   $PY /mnt/scfx_ly_run/scripts/dt_kvswap.py > /mnt/scfx_ly_run/logs/dtk_$3_$2.log 2>&1 < /dev/null &); echo "$(date -u) chain_stage4: launched $3 worker=$2 GPU=$1 N=$4" | tee -a "$LOG"; }
launch 0 dtk0 dtk_prompt_swapout 20
launch 1 dtk1 dtk_prompt_swapout 20
launch 2 dtk2 dtk_prompt_swapout 20
launch 3 dtk3 dtk_filler_swapin 20
launch 4 dtk4 dtk_filler_swapin 20
launch 5 dtk5 dtk_filler_swapin 20
launch 6 dtk6 dtk_prompt_swapctrl 20
launch 7 dtk7 dtk_prompt_swapctrl 20
# dtk_filler (plain filler text, no swap) is covered by chain_stage4b.sh after
# these finish -- the 2x2's text-only control needs fewer GPUs than the causal cells.
echo "$(date -u) chain_stage4: Stage 4 launched (8 workers)" | tee -a "$LOG"
