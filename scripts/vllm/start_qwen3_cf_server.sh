#!/usr/bin/env bash

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-4}
MAX_LEN=${MAX_LEN:-8192}
API_KEY=${API_KEY:-EMPTY}
GPU_UTIL=${GPU_UTIL:-0.92}

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --enable-prefix-caching \
  --generation-config vllm \
  --gpu-memory-utilization "$GPU_UTIL"
