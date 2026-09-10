#!/usr/bin/env bash
# stop|status for the rejudge passes. Lives in a file so the pattern never appears in an ssh command line.
PAT="scripts/rejudge_phase.py"
case "${1:-status}" in
  stop) pkill -f "$PAT"; sleep 3; pgrep -f "$PAT" >/dev/null && echo "rejudge still running" || echo "rejudge stopped";;
  status) for p in $(pgrep -f "$PAT"); do echo "pid $p shard=$(tr '\0' '\n' < /proc/$p/environ | grep ^SCFX_REJUDGE_SHARD= | cut -d= -f2) route=$(tr '\0' '\n' < /proc/$p/environ | grep -q '^OPENAI_PREFER_AZURE=false' && echo openai || echo azure) age=$(ps -o etimes= -p $p | tr -d ' ')s"; done;;
esac
