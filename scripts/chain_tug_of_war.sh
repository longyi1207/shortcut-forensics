#!/usr/bin/env bash
# VM-side scheduler: when both prompt_channel causal cells reach n>=12 ok rows,
# stop their workers and launch the tug-of-war cell on all 8 GPUs.
# Runs under nohup on the VM so it does not depend on the operator session.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
CAP=${CAP:-12}
c() { grep "\"phase\": \"prompt_channel\", \"condition\": \"$1\"" "$R" | grep -c '"status": "ok"'; }
while true; do
  a=$(c pc_base_add_delta26); b=$(c pc_prompt_ablate_delta26)
  if [ "$a" -ge "$CAP" ] && [ "$b" -ge "$CAP" ]; then break; fi
  sleep 120
done
echo "$(date -u) causal cells at $a/$b (cap $CAP) -> stopping pc1-8, launching tug-of-war"
for p in $(pgrep -f 'prompt_channe[l]'); do
  w=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^SCFX_WORKER_ID=' | cut -d= -f2)
  case "$w" in pc[1-8]) echo "stop $w pid=$p"; kill "$p";; esac
done
sleep 25
for g in 0 1 2 3 4 5 6 7; do
  (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$g SCFX_WORKER_ID=tw$g SCFX_PC_CONDITION=pc_prompt_add_tedium19 \
     SCFX_PC_PROMPT=1 SCFX_PC_VECS=tedium SCFX_PC_MODE=add SCFX_PC_LAYER=19 SCFX_PC_ALPHA=1.0 SCFX_PC_N=30 \
     /mnt/scfx_ly_run/.venv/bin/python /mnt/scfx_ly_run/scripts/prompt_channel.py > /mnt/scfx_ly_run/logs/pc_tug_tw$g.log 2>&1 < /dev/null &)
done
echo "$(date -u) tug-of-war launched: pc_prompt_add_tedium19, 8 workers, n=30"
