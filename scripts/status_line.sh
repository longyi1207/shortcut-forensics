#!/usr/bin/env bash
# One-line progress summary for the local Monitor (called over ssh every 15 min).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
workers=$(pgrep -fc '^(/mnt/scfx_ly_run/.venv/bin/)?python3? .*scripts/(b5_prompt_decay|prompt_channel|dt_capture|dt_mask|dt_heads|dt_kvswap)\.py')
tb=$(grep -l 'Traceback\|CUDA out of memory' logs/pc_*.log logs/dt_*.log logs/dtm_*.log logs/dth_*.log logs/dtmh_*.log 2>/dev/null | wc -l)
gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>1000{n++} END{print n+0}')
chain=$(pgrep -f '^bash (/mnt/scfx_ly_run/)?scripts/(chain_|run_mask)' | wc -l)
mask=$(cat logs/attn_mask_check.log outputs/20260821-launch/logs/attn_mask_check_heads.log 2>/dev/null | grep -o 'MASK_CHECK_[A-Z]*' | tail -1)
last=$(tail -1 logs/pipeline.log 2>/dev/null | cut -c1-90)
disk=$(df -h / | awk 'NR==2{print $5}')/$(df -h /mnt | awk 'NR==2{print $5}')
echo "pc tug=$(c prompt_channel pc_prompt_add_tedium19)/30 | dt base=$(c dt_capture dt_baseline)/20 prompt=$(c dt_capture dt_prompt)/20 | dtm p=$(c dt_mask dtm_prompt)/12 b=$(c dt_mask dtm_baseline)/12 mi=$(c dt_mask dtm_prompt_mask_instr)/20 mn=$(c dt_mask dtm_prompt_mask_notes)/20 mb=$(c dt_mask dtm_prompt_mask_both)/20 | dth p=$(c dt_heads dth_prompt)/12 b=$(c dt_heads dth_baseline)/6 | dtmh top8=$(c dt_mask dtm_prompt_mask_instr@top8)/20 rand8=$(c dt_mask dtm_prompt_mask_instr@rand8_s0)/20 top16=$(c dt_mask dtm_prompt_mask_instr@top16)/20 rand16=$(c dt_mask dtm_prompt_mask_instr@rand16_s0)/20 | workers=$workers gpus_busy=$gpus chains=$chain maskcheck=${mask:-pending} tb_logs=$tb disk=$disk | last: $last"
