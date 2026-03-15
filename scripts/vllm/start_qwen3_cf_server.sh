#!/usr/bin/env bash

set -euo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ -n "${VLLM_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODEL=${MODEL:-/data/home/vkropoti/models/Qwen3.5-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL}}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-1}
MAX_LEN=${MAX_LEN:-4096}
API_KEY=${API_KEY:-EMPTY}
GPU_UTIL=${GPU_UTIL:-0.92}
DTYPE=${DTYPE:-auto}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-0}
CALCULATE_KV_SCALES=${CALCULATE_KV_SCALES:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}
QUANTIZATION=${QUANTIZATION:-}
STRUCTURED_OUTPUTS_BACKEND=${STRUCTURED_OUTPUTS_BACKEND:-guidance}

echo "[vllm] CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "[vllm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[vllm] MODEL=${MODEL}"
echo "[vllm] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[vllm] TP=${TP}"
echo "[vllm] MAX_LEN=${MAX_LEN}"
echo "[vllm] DTYPE=${DTYPE}"
echo "[vllm] KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "[vllm] STRUCTURED_OUTPUTS_BACKEND=${STRUCTURED_OUTPUTS_BACKEND}"

cmd=(
  vllm serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --dtype "$DTYPE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
  --structured-outputs-config.backend "$STRUCTURED_OUTPUTS_BACKEND"
  --enable-prefix-caching
  --generation-config vllm
  --gpu-memory-utilization "$GPU_UTIL"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  cmd+=(--enable-chunked-prefill)
fi
if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  cmd+=(--async-scheduling)
fi
if [[ "${CALCULATE_KV_SCALES}" == "1" ]]; then
  cmd+=(--calculate-kv-scales)
fi
if [[ -n "${QUANTIZATION}" ]]; then
  cmd+=(--quantization "$QUANTIZATION")
fi

exec "${cmd[@]}"
