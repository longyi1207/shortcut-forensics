#!/usr/bin/env bash
# Persistent work queue: keeps the GPUs busy for as long as there is queued work.
#
# WHY THIS EXISTS: on 2026-08-31 the last chain and balancer exited when their
# targets were met, and because every follow-on experiment existed only in my
# notes rather than as an armed VM-side job, all 8 GPUs sat idle for ~14 hours.
# The chains solve "hand off from stage A to stage B"; nothing solved "the whole
# pipeline finished and there is more work on the list". This does.
#
# Reads jobs from queue.txt, one per line (blank lines and # comments ignored):
#     <script.py> <phase> <condition> <target_n>
# and keeps launching workers, newest-highest-gap first, whenever a GPU is free.
# A job is done when its condition reaches target_n ok rows. Exits only when
# every job is done AND queue.txt has not grown -- so appending a line to
# queue.txt is enough to give the cluster more work, from any session.
cd /mnt/scfx_ly_run || exit 1
R=outputs/20260821-launch/rollouts.jsonl
Q=${Q:-queue.txt}
LOG=logs/pipeline.log
PY=/mnt/scfx_ly_run/.venv/bin/python
POLL=${POLL:-180}
FREE_MB=${FREE_MB:-26000}
rows() { grep "\"phase\": \"$2\", \"condition\": \"$3\"" "$R" | grep -c '"status": "ok"'; }
envvars() {  # script -> "CONDVAR NVAR TAG"
  case "$1" in
    dt_kvswap.py)      echo "SCFX_DTK_CONDITION SCFX_DTK_N dtk" ;;
    prompt_channel.py) echo "SCFX_PC_CONDITION SCFX_PC_N pc" ;;
    dt_heads.py)       echo "SCFX_DTH_CONDITION SCFX_DTH_N dth" ;;
    dt_mask.py)        echo "SCFX_DTM_CONDITION SCFX_DTM_N dtm" ;;
    dt_capture.py)     echo "SCFX_DT_CONDITION SCFX_DT_N dt" ;;
    *) echo "" ;;
  esac
}
running_for() {  # count live workers for script $1 with condition $2
  local n=0 cv; cv=$(envvars "$1" | awk '{print $1}')
  for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/$1"); do
    v=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep "^$cv=" | cut -d= -f2)
    [ "$v" = "$2" ] && n=$((n+1))
  done
  echo "$n"
}
gpus_in_use() {
  for pid in $(pgrep -f "^$PY /mnt/scfx_ly_run/scripts/"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2
  done
}
echo "$(date -u) work_queue: started on $Q (poll ${POLL}s)" | tee -a "$LOG"
launched=0
while true; do
  [ -f "$Q" ] || { echo "$(date -u) work_queue: no $Q; exiting" | tee -a "$LOG"; exit 0; }
  best=""; best_gap=0
  while read -r script phase cond target _rest; do
    case "$script" in ""|\#*) continue ;; esac
    [ -z "$target" ] && continue
    have=$(rows "$script" "$phase" "$cond"); live=$(running_for "$script" "$cond")
    gap=$(( target - have - live ))
    if [ "$gap" -gt "$best_gap" ]; then best_gap=$gap; best="$script $phase $cond $target"; fi
  done < "$Q"
  if [ -z "$best" ]; then
    echo "$(date -u) work_queue: all queued jobs satisfied; sleeping (append to $Q for more work)" | tee -a "$LOG"
    sleep "$POLL"; continue
  fi
  set -- $best; script=$1; phase=$2; cond=$3; target=$4
  read -r cv nv tag <<< "$(envvars "$script")"
  mapfile -t used < <(gpus_in_use)
  for g in $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v m="$FREE_MB" '$2>m{print $1}'); do
    skip=0; for u in "${used[@]}"; do [ "$u" = "$g" ] && skip=1; done
    [ "$skip" = 1 ] && continue
    launched=$((launched+1)); wid="${tag}q${launched}"
    (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$g SCFX_WORKER_ID=$wid $cv=$cond $nv=$target \
       $PY /mnt/scfx_ly_run/scripts/$script > /mnt/scfx_ly_run/logs/${tag}_${cond}_${wid}.log 2>&1 < /dev/null &)
    echo "$(date -u) work_queue: GPU $g free -> $cond (gap $best_gap) worker=$wid" | tee -a "$LOG"
    break
  done
  sleep "$POLL"
done
