#!/usr/bin/env bash
# stop|status for the vLLM servers. Lives in a file so the pattern never appears in an ssh command line.
PAT="vllm serve"
case "${1:-status}" in
  stop) pkill -f "$PAT"; sleep 8; pgrep -f "$PAT" >/dev/null && { pkill -9 -f "$PAT"; sleep 3; }; pgrep -f "$PAT" >/dev/null && echo "vllm still running" || echo "vllm stopped";;
  status) for p in $(pgrep -f "$PAT"); do echo "pid $p gpu=$(tr '\0' '\n' < /proc/$p/environ | grep ^CUDA_VISIBLE_DEVICES= | cut -d= -f2)"; done;;
esac
