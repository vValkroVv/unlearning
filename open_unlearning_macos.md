# OpenUnlearning on macOS (Local CPU/MPS, Offline)

This document records what was set up and validated in this repo for local macOS usage, based on:

- [open_unlearning_venv_setup.md](open_unlearning_venv_setup.md)
- [part1_openlearning_myrun.md](part1_openlearning_myrun.md)
- [part1_openunlearning_runbook.md](part1_openunlearning_runbook.md)
- [scripts/fix_tofu_cache_datasets_3_0_1.py](scripts/fix_tofu_cache_datasets_3_0_1.py)

## Paths used

- Repo root: `/Users/valerii.kropotin/НОД/Diploma/open-unlearning`
- Venv: `./.venv`
- Model zip source: `offline_zips/zips/tofu_Llama-3.2-1B-Instruct_full.zip`
- Dataset zip source: `offline_zips/zips/TOFU_hf_cache.zip`
- Extracted model: `offline_zips/models/tofu_Llama-3.2-1B-Instruct_full`
- Extracted HF cache root: `offline_zips/data`
- HF hub cache: `offline_zips/data/hub`
- HF datasets cache: `offline_zips/data/datasets`

## 1) Python environment recreated

Executed:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -V
```

Observed Python: `3.12.7` (acceptable because `setup.py` requires `>=3.11`).

## 2) Dependencies installed for macOS (no FlashAttention)

What was done:

- Installed base tooling (`pip`, `setuptools`, `wheel`).
- Installed PyTorch stack pinned to repo expectation:
  - `torch==2.4.1`
  - `torchvision==0.19.1`
  - `torchaudio==2.4.1`
- Installed Hydra deps from `wheelhouse_hydra` + index resolution.
- Installed project dependencies with macOS adjustments:
  - skipped `flash-attn`
  - skipped `bitsandbytes` (not available for this env)
- Installed project editable mode with `--no-deps` to avoid failing hard dependency on bitsandbytes:
  - `pip install -e . --no-deps`
- Installed `lm-eval==0.4.8`.
- Installed `deepspeed==0.15.4` so `train.py`/`eval.py` imports succeed.

Current relevant package versions:

- `open-unlearning==0.1.0` (editable)
- `torch==2.4.1`
- `transformers==4.45.1`
- `datasets==3.0.1`
- `accelerate==0.34.2`
- `hydra-core==1.3.0`
- `hydra-colorlog==1.2.0`
- `deepspeed==0.15.4`
- `lm-eval==0.4.8`

Known remaining package warning:

- `pip check` reports `open-unlearning 0.1.0 requires bitsandbytes, which is not installed.`
- This is expected for this macOS flow.

## 3) Sanity checks completed

Executed and passed:

- `python -m compileall -q src`
- `python -c "import transformers, datasets; ..."`
- `python src/train.py --help`
- `python src/eval.py --help`

Runtime characteristics:

- `torch.cuda.is_available() == False`
- `torch.backends.mps.is_available() == True`

FlashAttention confirmation:

- `flash_attn` import: not installed
- `flash_attn_interface` import: not installed

## 4) Offline assets extracted

Executed:

```bash
unzip -o offline_zips/zips/tofu_Llama-3.2-1B-Instruct_full.zip -d offline_zips/models
unzip -o offline_zips/zips/TOFU_hf_cache.zip -d offline_zips/data
```

Validated outputs:

- `offline_zips/models/tofu_Llama-3.2-1B-Instruct_full/model.safetensors` exists
- `offline_zips/data/hub/datasets--locuslab--TOFU/...` exists
- `offline_zips/data/datasets/locuslab___tofu/...` exists

Approx sizes:

- model folder: `~2.3G`
- hub cache: `~12M`
- datasets cache: `~5.7M`

## 5) TOFU cache compatibility fix for `datasets==3.0.1`

As described in section 9 of [open_unlearning_venv_setup.md](open_unlearning_venv_setup.md), executed:

```bash
source .venv/bin/activate
python scripts/fix_tofu_cache_datasets_3_0_1.py \
  --datasets-cache "$PWD/offline_zips/data/datasets" \
  --dataset-dir-name locuslab___tofu
```

Patch summary:

- `dataset_info_files_scanned`: 18
- `arrow_files_scanned`: 18
- `dataset_info_files_patched`: 6
- `arrow_files_patched`: 6

Offline validation passed:

```bash
HF_HOME="$PWD/offline_zips/data" \
HF_DATASETS_CACHE="$PWD/offline_zips/data/datasets" \
HF_HUB_CACHE="$PWD/offline_zips/data/hub" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python - <<'PY'
from datasets import load_dataset
for cfg in ["forget10_perturbed", "forget10", "holdout10", "retain90"]:
    ds = load_dataset("locuslab/TOFU", name=cfg, split="train")
    print(cfg, len(ds))
PY
```

Observed:

- `forget10_perturbed 400`
- `forget10 400`
- `holdout10 400`
- `retain90 3600`

## 6) CPU eval plan for section 5 command (SDPA instead of FA2)

Reference section in [part1_openlearning_myrun.md](part1_openlearning_myrun.md): section `5) Final working eval command`.

For macOS local run, required override changes:

- `model.model_args.attn_implementation=sdpa`
- `model.model_args.device_map=cpu` (default config uses CUDA)
- `model.model_args.torch_dtype=float32`
- keep `local_files_only=true` for model/tokenizer
- keep `retain_logs_path` for reference metrics

Validated via Hydra compose dry-check (`--cfg job --resolve`) with no override conflicts.

Recommended command:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning
source .venv/bin/activate

HF_HOME=offline_zips/data \
HF_HUB_DISABLE_TELEMETRY=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_CACHE=offline_zips/data/hub \
HF_DATASETS_CACHE=offline_zips/data/datasets \
HYDRA_FULL_ERROR=1 \
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 holdout_split=holdout10 \
  model.model_args.attn_implementation=sdpa \
  model.model_args.device_map=cpu \
  model.model_args.torch_dtype=float32 \
  model.model_args.pretrained_model_name_or_path=offline_zips/models/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path=offline_zips/models/tofu_Llama-3.2-1B-Instruct_full \
  +model.tokenizer_args.local_files_only=true \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  eval.tofu.overwrite=true \
  eval.tofu.batch_size=2 \
  task_name=EVAL_tofu_full_Llama-3.2-1B-Instruct_TOFU_forget10_with_ref_cpu
```

## 7) Notes from real run logs

Your logs confirm the eval is working correctly in offline mode:

- evaluator started and wrote outputs under `saves/eval/EVAL_tofu_full_Llama-3.2-1B-Instruct_TOFU_forget10_with_ref_cpu`
- datasets loaded from local cache (`offline_zips/data/datasets/locuslab___tofu/...`)
- reference logs loaded from `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json`
- metric loops are running (e.g., `forget_Q_A_PARA_Prob`, `forget_Q_A_PERT_Prob`)

Why it is slower than H100:

- You are running CPU (`device_map=cpu`, `float32`, no CUDA kernels, no FlashAttention)
- H100 run used GPU acceleration and optimized attention kernels
- This slowdown is expected and not a conflict/error signal

## 8) Training smoke test on full model (CPU + SDPA)

Reference section in [part1_openlearning_myrun.md](part1_openlearning_myrun.md): section `7) Unlearning training runs`.

Goal was to run **training only** (no evaluation), with full model (not LoRA), locally on macOS.

### 8.1 Initial issue: MPS OOM

When running with only `device_map=cpu`, PyTorch still used MPS backend in this environment and crashed with:

- `RuntimeError: MPS backend out of memory ...`

### 8.2 Fixes applied

To make the smoke run stable:

- force CPU in trainer args:
  - `+trainer.args.use_cpu=true`
  - `+trainer.args.no_cuda=true`
- keep full model + SDPA:
  - `model.model_args.attn_implementation=sdpa`
  - `+model.model_args.device_map=cpu`
  - `model.model_args.torch_dtype=float32`
- disable eval path:
  - `trainer.args.do_eval=false`
  - `trainer.args.eval_on_start=false`
  - `trainer.args.eval_strategy=no`
- reduce sequence length for train datasets:
  - `data.forget.TOFU_QA_forget.args.max_length=128`
  - `data.retain.TOFU_QA_retain.args.max_length=128`
- set short smoke run:
  - `+trainer.args.max_steps=5`
- set `trainer.args.warmup_epochs=0` (important for CPU mode due warmup step computation using CUDA device count).

### 8.3 Working full-model CPU smoke command

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning
source .venv/bin/activate

HF_HOME=offline_zips/data \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_CACHE=offline_zips/data/hub \
HF_DATASETS_CACHE=offline_zips/data/datasets \
HYDRA_FULL_ERROR=1 \
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=GradAscent \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  model.model_args.attn_implementation=sdpa \
  +model.model_args.device_map=cpu \
  model.model_args.torch_dtype=float32 \
  model.model_args.pretrained_model_name_or_path=offline_zips/models/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path=offline_zips/models/tofu_Llama-3.2-1B-Instruct_full \
  +model.tokenizer_args.local_files_only=true \
  trainer.args.optim=adafactor \
  trainer.args.bf16=false \
  trainer.args.bf16_full_eval=false \
  trainer.args.warmup_epochs=0 \
  trainer.args.do_eval=false \
  trainer.args.eval_on_start=false \
  trainer.args.eval_strategy=no \
  +trainer.args.use_cpu=true \
  +trainer.args.no_cuda=true \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  data.forget.TOFU_QA_forget.args.max_length=128 \
  data.retain.TOFU_QA_retain.args.max_length=128 \
  +trainer.args.max_steps=5 \
  trainer.args.logging_steps=1 \
  retain_logs_path=null \
  task_name=UNLEARN_GradAscent_Llama-3.2-1B-Instruct_forget10_cpu_smoke
```

### 8.4 Smoke run status

Observed successful completion of 5 training steps with logs like:

- step losses/grad norms printed each step
- final summary contained `train_runtime`, `train_steps_per_second`, and `train_loss`

This confirms training path works locally for smoke testing.

## 9) Practical caveats

- This repo path contains non-ASCII characters (`НОД`).
- Hydra overrides with absolute paths may fail unless quoted.
- Use relative paths in overrides where possible.
- If absolute path is needed, quote override values, for example:
  - `"model.model_args.pretrained_model_name_or_path='/abs/path'"`.

## 10) Cleanup performed

Removed temporary outputs created during local CPU testing:

- `saves/unlearn/UNLEARN_GradAscent_Llama-3.2-1B-Instruct_forget10_cpu_smoke`
- `saves/eval/EVAL_tofu_full_Llama-3.2-1B-Instruct_TOFU_forget10_with_ref_cpu`
