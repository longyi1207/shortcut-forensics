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
#     <script.py> <phase> <condition> <target_n> [KEY=VAL,KEY=VAL,...]
# The optional 5th field carries extra environment for cells that need more than
# a condition name -- e.g. the steering cells need SCFX_PC_VECS / MODE / LAYER /
# ALPHA / PROMPT. Without it the queue could only launch env-free conditions.
# Workers are launched highest-gap-first whenever a GPU is free.
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
    dt_gdnswap.py)     echo "SCFX_DTK_CONDITION SCFX_DTK_N dtk" ;;
    *) echo "" ;;
  esac
}
running_for() {  # count live workers for script $1 with condition $2
  local n=0 cv; cv=$(envvars "$1" | awk '{print $1}')
  for pid in $(pgrep -f "scripts/$1"); do
    v=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep "^$cv=" | cut -d= -f2)
    [ "$v" = "$2" ] && n=$((n+1))
  done
  echo "$n"
}
gpus_in_use() {
  # Match BOTH absolute and relative script paths: dp_patch.py was launched as
  # `python scripts/dp_patch.py`, the absolute-only pattern missed it, the daemon
  # thought its GPU was free and put a rollout worker on top of it -> OOM.
  # Match on the SCRIPT PATH ONLY. Anchoring on $PY fails whenever a job was
  # started with a relative interpreter path (`.venv/bin/python scripts/x.py`),
  # which is how dp_patch was launched -- the daemon then judged its GPU free and
  # stacked a rollout worker on top of it. Twice.
  for pid in $(pgrep -f "scripts/[a-z_]*\.py"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2
  done
}
# single-instance guard: a second daemon double-books GPUs
LOCK=/tmp/scfx_work_queue.lock
exec 9>"$LOCK"
if ! flock -n 9; then echo "$(date -u) work_queue: another daemon holds $LOCK; exiting" | tee -a "$LOG"; exit 0; fi
echo "$(date -u) work_queue: started on $Q (poll ${POLL}s, lock $LOCK)" | tee -a "$LOG"
launched=0
# Failure backoff. A job whose worker dies on startup (bad vector file, typo in a
# env var) otherwise gets relaunched every poll forever: on 2026-09-01 a missing
# key in an .npz produced 134 relaunches of one cell while all 8 GPUs sat idle.
# Track rows-at-first-launch per condition; after MAX_TRIES launches with no new
# completed row, blacklist the job and say so loudly.
MAX_TRIES=${MAX_TRIES:-3}
declare -A TRIES BASELINE_ROWS DEAD
while true; do
  [ -f "$Q" ] || { echo "$(date -u) work_queue: no $Q; exiting" | tee -a "$LOG"; exit 0; }
  best=""; best_gap=0; best_env=""
  while read -r script phase cond target extra _rest; do
    case "$script" in ""|\#*) continue ;; esac
    [ -z "$target" ] && continue
    [ -n "${DEAD[$cond]}" ] && continue
    have=$(rows "$script" "$phase" "$cond"); live=$(running_for "$script" "$cond")
    # a job that has been launched MAX_TRIES times and produced no new row is broken
    if [ -n "${TRIES[$cond]}" ] && [ "${TRIES[$cond]}" -ge "$MAX_TRIES" ] && [ "$have" -le "${BASELINE_ROWS[$cond]}" ] && [ "$live" -eq 0 ]; then
      DEAD[$cond]=1
      echo "$(date -u) work_queue: !! BLACKLISTING $cond -- ${TRIES[$cond]} launches, still $have rows. Its workers are dying on startup; check logs/${tag}_${cond}_*.log" | tee -a "$LOG"
      continue
    fi
    gap=$(( target - have - live ))
    if [ "$gap" -gt "$best_gap" ]; then best_gap=$gap; best="$script $phase $cond $target"; best_env="$extra"; fi
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
    [ -z "${TRIES[$cond]}" ] && { TRIES[$cond]=0; BASELINE_ROWS[$cond]=$(rows "$script" "$phase" "$cond"); }
    TRIES[$cond]=$(( ${TRIES[$cond]} + 1 ))
    launched=$((launched+1)); wid="${tag}q${launched}"
    extra_env=""
    [ -n "$best_env" ] && extra_env=$(echo "$best_env" | tr ',' ' ')
    (cd /mnt/scfx_ly_run && setsid nohup env CUDA_VISIBLE_DEVICES=$g SCFX_WORKER_ID=$wid $cv=$cond $nv=$target $extra_env \
       $PY /mnt/scfx_ly_run/scripts/$script > /mnt/scfx_ly_run/logs/${tag}_${cond}_${wid}.log 2>&1 < /dev/null &)
    echo "$(date -u) work_queue: GPU $g free -> $cond (gap $best_gap) worker=$wid ${best_env:+env=$best_env}" | tee -a "$LOG"
    sleep 25          # let it claim the GPU before the next free-memory scan
    continue 2        # re-evaluate gaps, then fill the NEXT free GPU in this same pass
  done
  sleep "$POLL"
done
