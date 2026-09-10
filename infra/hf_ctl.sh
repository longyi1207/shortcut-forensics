#!/usr/bin/env bash
# stop|status for the HF rollout workers (run_phase.py / prompt_channel.py). Pattern lives in a file so it never
# appears in an ssh command line. Rows are appended per finished rollout, so stopping loses only in-flight ones.
PAT="scripts/(run_phase|prompt_channel)\.py"
case "${1:-status}" in
  stop) pkill -f "$PAT"; sleep 8; pgrep -f "$PAT" >/dev/null && { pkill -9 -f "$PAT"; sleep 3; }; pgrep -f "$PAT" >/dev/null && echo "hf workers still running" || echo "hf workers stopped";;
  status) for p in $(pgrep -f "$PAT"); do echo "pid $p $(tr '\0' '\n' < /proc/$p/environ | grep -E '^(CUDA_VISIBLE_DEVICES|SCFX_WORKER_ID)=' | tr '\n' ' ') age=$(ps -o etimes= -p $p | tr -d ' ')s"; done;;
esac
