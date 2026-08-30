#!/usr/bin/env bash
# Keep every GPU busy for the rest of a phase.
#
# Problem this solves: workers are pinned to one condition and exit when THAT
# condition reaches its target. With 8 GPUs and 3 cells of n=20 the split is
# 3/3/2, so the two-worker cell needs 10 rollouts per worker (~10 h) while the
# others finish in ~7 h -- six GPUs would sit idle for hours.
#
# This loop polls every POLL seconds; whenever a GPU is free and some condition
# still needs more rollouts than it has running workers, it launches one more
# worker for whichever condition is furthest behind. It never exceeds a
# condition's target (completed rows + live workers <= target).
#
# Usage: gpu_balancer.sh <script.py> <phase> "cond:target cond:target ..."
#   e.g. gpu_balancer.sh dt_kvswap.py dt_kvswap "dtk_prompt_swapout:20 dtk_filler_swapin:20 dtk_prompt_swapctrl:20"
# Exits when every condition has reached its target.
cd /mnt/scfx_ly_run || exit 1
SCRIPT="$1"; PHASE="$2"; SPEC="$3"
[ -z "$SPEC" ] && { echo "usage: gpu_balancer.sh <script.py> <phase> \"cond:target ...\""; exit 2; }
R=outputs/20260821-launch/rollouts.jsonl
LOG=logs/pipeline.log
PY=/mnt/scfx_ly_run/.venv/bin/python
POLL=${POLL:-300}
FREE_MB=${FREE_MB:-26000}
case "$SCRIPT" in
  dt_kvswap.py)      CVAR=SCFX_DTK_CONDITION; NVAR=SCFX_DTK_N; TAG=dtk ;;
  prompt_channel.py) CVAR=SCFX_PC_CONDITION;  NVAR=SCFX_PC_N;  TAG=pc  ;;
  dt_heads.py)       CVAR=SCFX_DTH_CONDITION; NVAR=SCFX_DTH_N; TAG=dth ;;
  dt_mask.py)        CVAR=SCFX_DTM_CONDITION; NVAR=SCFX_DTM_N; TAG=dtm ;;
  *) echo "unknown script $SCRIPT"; exit 2 ;;
esac
rows() { grep "\"phase\": \"$PHASE\", \"condition\": \"$1\"" "$R" | grep -c '"status": "ok"'; }
running() {  # live workers whose CVAR env equals $1
  local n=0
  for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/$SCRIPT"); do
    v=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep "^$CVAR=" | cut -d= -f2)
    [ "$v" = "$1" ] && n=$((n+1))
  done
  echo "$n"
}
busy_gpus() {  # GPU indices already hosting one of OUR workers (via CUDA_VISIBLE_DEVICES)
  for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2
  done
}
echo "$(date -u) balancer[$PHASE]: watching '$SPEC' (poll ${POLL}s, needs ${FREE_MB}MB free)" | tee -a "$LOG"
launched=0
while true; do
  done_all=1; worst=""; worst_gap=0
  for item in $SPEC; do
    cond=${item%%:*}; target=${item##*:}
    have=$(rows "$cond"); live=$(running "$cond")
    [ "$have" -ge "$target" ] || done_all=0
    gap=$(( target - have - live ))
    if [ "$gap" -gt "$worst_gap" ]; then worst_gap=$gap; worst=$cond; fi
  done
  [ "$done_all" = 1 ] && { echo "$(date -u) balancer[$PHASE]: all targets reached; launched $launched extra workers; exiting" | tee -a "$LOG"; exit 0; }
  if [ -n "$worst" ]; then
    mapfile -t used < <(busy_gpus)
    for g in $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v m="$FREE_MB" '$2>m{print $1}'); do
      skip=0; for u in "${used[@]}"; do [ "$u" = "$g" ] && skip=1; done
      [ "$skip" = 1 ] && continue
      launched=$((launched+1))
      wid="${TAG}bal${launched}"
      target=$(for i in $SPEC; do [ "${i%%:*}" = "$worst" ] && echo "${i##*:}"; done)
      (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$g SCFX_WORKER_ID=$wid $CVAR=$worst $NVAR=$target \
         $PY /mnt/scfx_ly_run/scripts/$SCRIPT > /mnt/scfx_ly_run/logs/${TAG}_${worst}_${wid}.log 2>&1 < /dev/null &)
      echo "$(date -u) balancer[$PHASE]: GPU $g idle -> launched $worst worker=$wid (gap was $worst_gap)" | tee -a "$LOG"
      break  # one launch per poll; re-evaluate gaps next round
    done
  fi
  sleep "$POLL"
done
