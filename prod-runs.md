# Production Runs (Default Script Hyperparams)

## Common setup

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
```

## 1) GA - DUET

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/duet/ga_duet.sh
```

## 2) GA - UNLamb

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/popqa/ga_popqa.sh
```

## 3) NPO - DUET

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/duet/npo_duet.sh
```

## 4) NPO - UNLamb

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/popqa/npo_popqa.sh
```

## 5) FALCON - DUET

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/duet/falcon_duet.sh
```

## 6) FALCON - UNLamb

```bash
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/popqa/falcon_popqa.sh
```
