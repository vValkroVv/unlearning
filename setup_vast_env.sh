#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_JOBS_VAL="${MAX_JOBS:-8}"

echo "[setup] repo_root=${REPO_ROOT}"
echo "[setup] creating or reusing virtualenv at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[setup] upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing core dependencies"
pip install \
  numpy==2.2.3 hydra-core==1.3.0 hydra-colorlog==1.2.0 omegaconf==2.3.0 \
  transformers==4.45.1 accelerate==0.34.2 datasets==3.0.1 peft==0.15.2 \
  deepspeed==0.15.4 scipy==1.14.1 tqdm==4.67.1 rouge-score==0.1.2 \
  scikit-learn==1.5.2 huggingface-hub==0.29.1 sentencepiece==0.2.1 \
  evaluate==0.4.3 lm-eval==0.4.8 jsonlines==4.0.0 pytorch-revgrad==0.2.0 \
  einops==0.8.1 pandas==2.3.0

echo "[setup] installing CUDA torch (cu124)"
pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo "[setup] pinning fsspec for compatibility"
pip install fsspec==2024.6.1

echo "[setup] installing flash-attn"
MAX_JOBS="${MAX_JOBS_VAL}" pip install --no-build-isolation flash-attn==2.6.3

echo "[setup] installing duet-adjacent runtime extras"
pip install bitsandbytes==0.44.1 tensorboard==2.20.0 wandb==0.21.0

export HF_HOME="${REPO_ROOT}/.hf_home"
export TRITON_CACHE_DIR="${REPO_ROOT}/.triton"
mkdir -p "${HF_HOME}/hub" "${TRITON_CACHE_DIR}"
echo "[setup] HF_HOME=${HF_HOME}"
echo "[setup] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

echo "[check] pip consistency"
pip check

echo "[check] import sweep under src/"
python - <<'PY'
import os
import sys
import importlib

src = "/workspace/unlearning/src"
sys.path.insert(0, src)
mods = []
for root, _, files in os.walk(src):
    for f in files:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(root, f), src)
            m = rel[:-3].replace("/", ".")
            if m.endswith(".__init__"):
                m = m[:-9]
            if m:
                mods.append(m)
mods = sorted(set(mods))
fails = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        fails.append((m, type(e).__name__, str(e)))
print("TOTAL_MODULES", len(mods))
print("FAILED", len(fails))
if fails:
    for m, et, msg in fails:
        print(f"{m}::{et}::{msg}")
    raise SystemExit(1)
PY

echo "[check] entry points"
python src/train.py --help >/dev/null
python src/eval.py --help >/dev/null
python setup_data.py --help >/dev/null

echo "[check] bitsandbytes"
python -m bitsandbytes >/dev/null

echo "[check] GPU + flash-attn smoke test"
python - <<'PY'
import torch
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_func

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

q = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 16, 8, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

print("flash_attn_version", getattr(flash_attn, "__version__", "unknown"))
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("flash_attn_out_shape", tuple(out.shape))
print("flash_attn_out_dtype", out.dtype)
print("gpu_name", torch.cuda.get_device_name(0))
PY

echo "[ready] setup completed successfully"
echo "[ready] run these in each new shell:"
echo "  cd ${REPO_ROOT}"
echo "  source .venv/bin/activate"
echo "  export HF_HOME=${REPO_ROOT}/.hf_home"
echo "  export TRITON_CACHE_DIR=${REPO_ROOT}/.triton"
echo "  mkdir -p \"\$HF_HOME/hub\" \"\$TRITON_CACHE_DIR\""
