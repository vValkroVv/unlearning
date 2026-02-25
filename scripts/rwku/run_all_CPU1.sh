#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
# export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4"}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"}

echo "[rwku] Running NPO"
bash "${script_dir}/npo_rwku.sh"

echo "[rwku] Running AdaWGD"
bash "${script_dir}/ada_wgd_rwku.sh"
