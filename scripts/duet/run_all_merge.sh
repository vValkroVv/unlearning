#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"}
export MERGE_POPULARITY_FORGET=1

echo "[duet] Running Adapop (merged forget)"
bash "${script_dir}/ada_pop_duet.sh"

echo "[duet] Running GA (merged forget)"
bash "${script_dir}/ga_duet.sh"

echo "[duet] Running WGA (merged forget)"
bash "${script_dir}/wga_duet.sh"

# echo "[duet] Running AdaWGD (merged forget)"
# bash "${script_dir}/ada_wgd_duet.sh"

echo "[duet] Running GD (merged forget)"
bash "${script_dir}/gd_duet.sh"

echo "[duet] Running NPO (merged forget)"
bash "${script_dir}/npo_duet.sh"
