#!/usr/bin/env bash
# One-line progress summary for the local Monitor (called over ssh every 15 min).
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
c() { grep "\"phase\": \"$1\", \"condition\": \"$2\"" "$R" | grep -c '"status": "ok"'; }
workers=$(pgrep -af 'b5_prompt_deca[y]|prompt_channe[l]|dt_captur[e]|dt_mas[k]\.py' | grep -vc 'bash -c')
tb=$(grep -l 'Traceback\|CUDA out of memory' logs/pc_*.log logs/dt_*.log logs/dtm_*.log 2>/dev/null | wc -l)
gpus=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>1000{n++} END{print n+0}')
chain=$(pgrep -f 'chain_stage[0-9a-z_]*\.sh|chain_tug_of_wa[r]|run_mask_check_when_fre[e]' | wc -l)
mask=$(grep -o 'MASK_CHECK_[A-Z]*' logs/attn_mask_check.log 2>/dev/null | tail -1)
echo "pc add=$(c prompt_channel pc_base_add_delta26) abl=$(c prompt_channel pc_prompt_ablate_delta26) tug=$(c prompt_channel pc_prompt_add_tedium19)/30 | dt base=$(c dt_capture dt_baseline)/20 prompt=$(c dt_capture dt_prompt)/20 | dtm p=$(c dt_mask dtm_prompt) b=$(c dt_mask dtm_baseline) mi=$(c dt_mask dtm_prompt_mask_instr) mn=$(c dt_mask dtm_prompt_mask_notes) mb=$(c dt_mask dtm_prompt_mask_both) | workers=$workers gpus_busy=$gpus chains=$chain maskcheck=${mask:-pending} | tb_logs=$tb"
