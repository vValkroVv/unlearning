#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"}
export BASE_MODEL="Qwen2.5-7B-Instruct"
export MODEL_CONFIG="Qwen2.5-7B-Instruct-lora"
export HF_BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

echo "[rwku][qwen2.5-7b] Running GA"
bash "${script_dir}/ga_rwku.sh"

echo "[rwku][qwen2.5-7b] Running GD"
bash "${script_dir}/gd_rwku.sh"

echo "[rwku][qwen2.5-7b] Running WGA"
bash "${script_dir}/wga_rwku.sh"

echo "[rwku][qwen2.5-7b] Running NPO"
bash "${script_dir}/npo_rwku.sh"

echo "[rwku][qwen2.5-7b] Running AdaPop"
bash "${script_dir}/ada_pop_rwku.sh"
