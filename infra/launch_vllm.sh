#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server per GPU for the prompt sweep.
# Usage (on the box): infra/launch_vllm.sh <gpu_csv>     e.g. 0,1,2,3
# Server for GPU g listens on port $((8123+g)). Logs: /mnt/scfx_logs/vllm_g<g>.log
# Context: real 80-turn rollouts reach 65k-120k tokens (WRITEUP §4.12), so the
# server is sized at 128k; the GDN hybrid has only 8 attention layers so the
# KV cache for 16 concurrent long sequences fits an 80GB A100.
set -euo pipefail
GPU_CSV="${1:?usage: launch_vllm.sh <gpu_csv>}"
VENV="${VLLM_VENV:-/mnt/scfx_vllm3}"
MODEL=Qwen/Qwen3.5-9B
mkdir -p /mnt/scfx_logs
IFS=',' read -ra GPUS <<< "$GPU_CSV"
for g in "${GPUS[@]}"; do
  port=$((8123 + g))
  echo "[launch_vllm] GPU $g -> port $port"
  setsid nohup env CUDA_VISIBLE_DEVICES="$g" HF_HOME=/mnt/scfx_ly_cache \
    PATH="$VENV/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    "$VENV/bin/vllm" serve "$MODEL" --port "$port" --host 127.0.0.1 \
      --dtype bfloat16 --max-model-len 131072 --max-num-seqs 16 \
      --gpu-memory-utilization 0.90 --trust-remote-code \
      > "/mnt/scfx_logs/vllm_g${g}.log" 2>&1 < /dev/null &
done
echo "[launch_vllm] started; poll with: curl -s localhost:<port>/health"
