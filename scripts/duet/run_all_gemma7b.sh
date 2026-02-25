#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"}
export BASE_MODEL="gemma-7b-it"
export MODEL_CONFIG="gemma-7b-it-lora"
export HF_BASE_MODEL_PATH="google/gemma-7b-it"
export LOCAL_SFT_BASE="/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/gemma-7b-it_full_3ep_ft_tripunlamb"
export USE_SFT_BASE=1
export MERGE_POPULARITY_FORGET=1

echo "[duet][gemma-7b] Running GA"
bash "${script_dir}/ga_duet.sh"

echo "[duet][gemma-7b] Running AdaPop"
bash "${script_dir}/ada_pop_duet.sh"

echo "[duet][gemma-7b] Running GD"
bash "${script_dir}/gd_duet.sh"

echo "[duet][gemma-7b] Running WGA"
bash "${script_dir}/wga_duet.sh"

echo "[duet][gemma-7b] Running NPO"
bash "${script_dir}/npo_duet.sh"
