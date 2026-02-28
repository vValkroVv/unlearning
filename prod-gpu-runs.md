# Production GPU Runs (Offline Server)

This is the same runbook as `prod-runs.md`, adapted for your offline server root:

- `/data/home/vkropoti/unlearning`
- local repos under `SwetieePawsss/...`

## Current Local Structure

Expected layout inside `/data/home/vkropoti/unlearning`:

```text
SwetieePawsss/
  DUET/
    data/*.parquet
  exp_UNLamb/
    data/*.parquet
  DUET_ft_models/
    llama-3.1-8b-instruct-tripunlamb-ft/
  UNLamb_ft_models/
    llama-3.1-8b-instruct-popqa-ft/
```

Notes:
- Keep `LOCAL_SFT_BASE` as repo-id-like local path root, and set `SFT_SUBFOLDER` to the model folder.
- Folder name must be exactly `SwetieePawsss` (not misspelled).

## One-time path wiring

If your code is in `/home/vkropoti/diploma/open-unlearning` but models/datasets are in
`/data/home/vkropoti/unlearning/SwetieePawsss`, create this symlink once:

```bash
ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss /home/vkropoti/diploma/open-unlearning/SwetieePawsss
```

## Common setup

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

# force offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Important:
- `HF_DATASETS_OFFLINE` must not contain spaces.
- Wrong: `export HF_DATASETS OFFLINE=1`
- Correct: `export HF_DATASETS_OFFLINE=1`

## 1) GA - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/ga_duet.sh
```

## 2) GA - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/ga_popqa.sh
```

## 3) NPO - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_duet.sh
```

## 4) NPO - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/npo_popqa.sh
```

## 5) FALCON - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
MI_SELECT_LAYERS=1 \
MI_MODEL_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MI_TOKENIZER_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
bash scripts/duet/falcon_duet.sh
```

## 6) FALCON - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
MI_SELECT_LAYERS=1 \
MI_MODEL_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MI_TOKENIZER_PATH=/data/home/vkropoti/unlearning/assets/tokenizers/llama-3.1-8b-instruct-chat-template \
bash scripts/popqa/falcon_popqa.sh
```
