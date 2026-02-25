#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"} 
#  5e-3 1e-2

echo "[duet] Running Ada_pop"
bash "${script_dir}/ada_pop_duet.sh"


echo "[duet] Running GA"
bash "${script_dir}/ga_duet.sh"

echo "[duet] Running GD"
bash "${script_dir}/gd_duet.sh"

echo "[duet] Running WGA"
bash "${script_dir}/wga_duet.sh"

echo "[duet] Running NPO"
bash "${script_dir}/npo_duet.sh"

# echo "[duet] Running AdaWGD"
# bash "${script_dir}/ada_wgd_duet.sh"
