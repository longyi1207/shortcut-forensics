#!/usr/bin/env bash
# One-line progress summary for the local Monitor (called over ssh every 15 min).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
workers=$(pgrep -fc '^(/mnt/scfx_ly_run/.venv/bin/)?python3? .*scripts/(b5_prompt_decay|prompt_channel|dt_capture|dt_mask|dt_heads|dt_kvswap).py')
tb=$(grep -l 'Traceback\|CUDA out of memory' logs/pc_*.log logs/dt_*.log logs/dtm_*.log logs/dth_*.log logs/dtmh_*.log 2>/dev/null | wc -l)
gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>1000{n++} END{print n+0}')
chain=$(pgrep -f '^bash (/mnt/scfx_ly_run/)?scripts/(chain_|run_mask)' | wc -l)
mask=$(cat logs/attn_mask_check.log outputs/20260821-launch/logs/attn_mask_check_heads.log 2>/dev/null | grep -o 'MASK_CHECK_[A-Z]*' | tail -1)
last=$(tail -1 logs/pipeline.log 2>/dev/null | cut -c1-90)
disk=$(df -h / | awk 'NR==2{print $5}')/$(df -h /mnt | awk 'NR==2{print $5}')
echo "STAGE4 swapout=$(c dt_kvswap dtk_prompt_swapout)/20 swapin=$(c dt_kvswap dtk_filler_swapin)/20 swapctrl=$(c dt_kvswap dtk_prompt_swapctrl)/20 filler=$(c dt_kvswap dtk_filler)/20 | dose a05=$(c prompt_channel pc_add_tedium19_a05)/20 p_a05=$(c prompt_channel pc_prompt_add_tedium19_a05)/20 | [done] tug=$(c prompt_channel pc_prompt_add_tedium19)/30 dt=$(c dt_capture dt_baseline)+$(c dt_capture dt_prompt) dtm=$(c dt_mask dtm_prompt_mask_instr)+$(c dt_mask dtm_prompt_mask_notes)+$(c dt_mask dtm_prompt_mask_both) dth=$(c dt_heads dth_prompt)+$(c dt_heads dth_baseline) | workers=$workers gpus_busy=$gpus chains=$chain kvswap=$(grep -o 'KV_SWAP_[A-Z]*' logs/kv_swap_check.log 2>/dev/null | tail -1) tb_logs=$tb disk=$disk | last: $last"
