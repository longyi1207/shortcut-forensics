#!/usr/bin/env bash
# Start the night guardian if it is not running; `restart` replaces it.
# Lives in a file so the kill/check patterns never appear in an ssh command line
# (pkill -f from an ssh one-liner matches the one-liner's own shell).
cd /mnt/scfx_ly_run || exit 1
export HF_HOME=/mnt/scfx_ly_cache PYTHONUNBUFFERED=1
PAT="python infra/guardian.py"
if [ "${1:-}" = restart ] || [ "${1:-}" = stop ]; then pkill -f "$PAT"; sleep 3; fi
if [ "${1:-}" = stop ]; then pgrep -f "$PAT" >/dev/null && echo "guardian still running" || echo "guardian stopped"; exit 0; fi
if pgrep -f "$PAT" > /dev/null; then echo "guardian running: $(pgrep -f "$PAT" | tr '\n' ' ')"; exit 0; fi
nohup setsid .venv/bin/python infra/guardian.py >> /mnt/scfx_logs/guardian.out 2>&1 < /dev/null &
echo "guardian launched pid $!"
