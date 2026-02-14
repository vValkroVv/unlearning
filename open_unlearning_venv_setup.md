# OpenUnlearning venv setup (H100 node, no public internet)

This note documents the exact venv setup flow that worked on the H100 node for `open-unlearning`, including the workaround for missing Hydra packages in an internal PyPI mirror.

> Assumptions
> - You **can access an internal PyPI mirror** from the H100 node (e.g., Artifactory).
> - The H100 node has **no access to public internet** (PyPI / Hugging Face).
> - You can use a separate machine with internet to download missing wheels and then transfer them to the H100 node.

---

## 0) Create and activate venv (Python 3.11)

```bash
cd /path/to/workdir

python3.11 -m venv unlearning-venv
source unlearning-venv/bin/activate

which python
python -V
```

Upgrade pip tooling (recommended):

```bash
python -m pip install -U pip setuptools wheel
python -m pip --version
```

---

## 1) Install PyTorch (from your internal index) and sanity-check CUDA

Install (example — use your org’s recommended command):

```bash
python -m pip install torch torchvision torchaudio
```

Sanity check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda runtime:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul mean:", y.mean().item())
PY
```

---

## 2) Clone repo (or `cd` into existing clone)

```bash
git clone <YOUR_INTERNAL_MIRROR_OR_PRECLONED_REPO_PATH>
cd open-unlearning
```

---

## 3) Detect which requirements are missing from the internal index (optional but useful)

This checks each *top-level* requirement line without pulling dependencies:

```bash
mkdir -p /tmp/pip_probe
: > /tmp/pip_probe.ok
: > /tmp/pip_probe.fail

while IFS= read -r line; do
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [ -z "$line" ] && continue
  case "$line" in
    -r*|--*|-e*|git+*|http:*|https:* ) continue ;;
  esac

  if python -m pip download --no-deps -d /tmp/pip_probe "$line" -q; then
    echo "OK   $line" | tee -a /tmp/pip_probe.ok
  else
    echo "FAIL $line" | tee -a /tmp/pip_probe.fail
  fi
done < requirements.txt

echo "FAIL list:"
cat /tmp/pip_probe.fail
```

In our case, only these were missing:
- `hydra-core==1.3`
- `hydra_colorlog==1.2.0` (package name may appear as `hydra-colorlog` on PyPI)

---

## 4) Fix missing Hydra packages via a mini wheelhouse (transfer from an internet machine)

### 4.1 On an internet machine: download a minimal wheelhouse

Create a folder with wheels for Hydra + dependencies and pack it:

```bash
python3.11 -m venv hydra-dl
source hydra-dl/bin/activate
python -m pip install -U pip

mkdir -p wheelhouse_hydra

# Prefer wheels only (avoid .tar.gz builds on the offline machine).
python -m pip download -d wheelhouse_hydra --only-binary :all:   "hydra-core==1.3.0"   "hydra-colorlog==1.2.0"   "omegaconf==2.3.0"   "antlr4-python3-runtime==4.9.3"   "PyYAML>=5.1"   "colorlog>=6.7.0"

tar -czf wheelhouse_hydra.tar.gz wheelhouse_hydra
```

Transfer `wheelhouse_hydra.tar.gz` to the H100 node (scp/rsync/USB/etc).

> Note: If `--only-binary :all:` fails for your platform, re-run without it:
> `pip download -d wheelhouse_hydra ...`
> (Then you may need a compiler toolchain on the H100 node to build sdists.)

### 4.2 On the H100 node: unpack + install Hydra from the wheelhouse

```bash
cd /path/to/open-unlearning
source ../unlearning-venv/bin/activate

tar -xzf wheelhouse_hydra.tar.gz
```

Install Hydra **fully offline** (from local wheelhouse only):

```bash
python -m pip install --no-index --find-links "$PWD/wheelhouse_hydra"   "hydra-core==1.3.0" "hydra-colorlog==1.2.0"
```

If you see an error about “installing build dependencies … setuptools not found”, it means pip picked an sdist and tried to build it in an isolated env. Quick fix:

```bash
python -m pip install --no-index --find-links "$PWD/wheelhouse_hydra"   --no-build-isolation   "hydra-core==1.3.0" "hydra-colorlog==1.2.0"
```

Sanity check (plugin import path is under `hydra_plugins`):

```bash
python - <<'PY'
import hydra
import hydra_plugins.hydra_colorlog
print("hydra:", hydra.__version__)
print("hydra-colorlog plugin: OK")
PY
```

---

## 5) Install the rest of repo dependencies (keep current environment)

Now that Hydra is installed, install the remaining `requirements.txt` using your internal index, but **also keep** the wheelhouse in the search path (so Hydra is always satisfiable).

> Important: do **not** pass `-U` unless you explicitly want upgrades/downgrades.

```bash
python -m pip install -r requirements.txt --find-links "$PWD/wheelhouse_hydra"
python -m pip check
```

(Optional) preview what pip *would* change before doing it:

```bash
python -m pip install --dry-run -r requirements.txt --find-links "$PWD/wheelhouse_hydra"
```

---

## 6) Install FlashAttention on H100 (recommended)

OpenUnlearning model configs for Llama/Qwen set:
`model_args.attn_implementation: flash_attention_2`.

This means:
- If you keep defaults, you need FA2 (`flash_attn` import path).
- If you install only FA3 (Hopper), pass a Hydra override at runtime:
  `model.model_args.attn_implementation=flash_attention_3`
  (or `sdpa` as fallback).

### 6.1 Pin to an H100 by UUID (avoid landing on L40S)

```bash
nvidia-smi -L
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-<YOUR_H100_UUID>
```

Sanity check:

```bash
python - <<'PY'
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))
PY
```

Expected: compute capability `(9, 0)`.

### 6.2 Ensure CUDA toolkit (`nvcc`) is available for source builds

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvcc -V
```

For FA3 hopper build, `nvcc` must be CUDA >= 12.3.

### 6.3 Build/install FA3 from full source tarball

Use the full archive that contains `flash-attention/hopper/flash_api.cpp` and `flash_*sm90*.cu`:

```bash
cd /path/to/open-unlearning
tar -xzf flash-attention-v2.6.3-full.tar.gz

python -m pip install -U ninja packaging psutil einops pytest

cd flash-attention/hopper
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
python setup.py install
```

Verify FA3 modules:

```bash
python - <<'PY'
import importlib.util
print("flash_attn_interface:", bool(importlib.util.find_spec("flash_attn_interface")))
print("flashattn_hopper_cuda:", bool(importlib.util.find_spec("flashattn_hopper_cuda")))
PY
```

### 6.4 Optional: also build/install FA2 to support default OpenUnlearning configs

If you want to run without overriding `attn_implementation`, install FA2 as well:

```bash
cd /path/to/open-unlearning/flash-attention
export MAX_JOBS=4
python setup.py install
```

Verify FA2 module:

```bash
python - <<'PY'
import importlib.util
print("flash_attn:", bool(importlib.util.find_spec("flash_attn")))
PY
```

---

## 7) Install OpenUnlearning in editable mode

```bash
# Canonical extra name:
python -m pip install -e ".[lm-eval]" --find-links "$PWD/wheelhouse_hydra"
# If your local setup.py only defines lm_eval, use:
# python -m pip install -e ".[lm_eval]" --find-links "$PWD/wheelhouse_hydra"
python -m pip check
```

---

## 8) Quick sanity checks

```bash
python -m compileall -q src
python -c "import transformers, datasets; print('transformers', transformers.__version__); print('datasets', datasets.__version__)"
python src/train.py --help | head -n 30
python src/eval.py  --help | head -n 30
```

If these pass, the venv is ready for the next stage (datasets/models/logs).

---

## 9) TOFU cache compatibility fix (`datasets==3.0.1`)

If loading `locuslab/TOFU` fails with:
`TypeError: must be called with a dataclass type or instance`,
your cached TOFU metadata likely uses `"_type": "List"` while `datasets==3.0.1` expects `Sequence`.

Run the cache patcher from repo root:

```bash
python scripts/fix_tofu_cache_datasets_3_0_1.py \
  --datasets-cache "$HF_DATASETS_CACHE" \
  --dataset-dir-name locuslab___tofu
```

Then verify:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from datasets import load_dataset
for cfg in ["forget10_perturbed", "forget10", "holdout10", "retain90"]:
    ds = load_dataset("locuslab/TOFU", name=cfg, split="train")
    print(cfg, len(ds))
PY
```
