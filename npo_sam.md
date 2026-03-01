# NPO-SAM Integration and Validation Notes

Date: 2026-03-01
Repo: `/workspace/unlearning`

## 1) What was validated

### 1.1 NPO-SAM trainer/config wiring

Checked that NPO-SAM is fully wired in code/config:

- Trainer implementation:
  - `src/trainer/unlearn/npo_sam.py`
- Trainer registration:
  - `src/trainer/__init__.py`
- Trainer config:
  - `configs/trainer/NPOSAM.yaml`
- Experiment configs:
  - `configs/experiment/unlearn/duet/npo_sam_lora.yaml`
  - `configs/experiment/unlearn/popqa/npo_sam_lora.yaml`

Verified method args are present and plumbed through scripts:

- `beta`, `alpha`, `gamma`
- `sam_rho`, `sam_adaptive`, `sam_eps`

### 1.2 NPO-SAM run scripts

Validated both target scripts end-to-end:

- `scripts/duet/npo_sam_duet.sh`
- `scripts/popqa/npo_sam_popqa.sh`

### 1.3 Production docs coverage

Added production launch commands for NPO-SAM to:

- `prod-runs.md`
  - `## 7) NPO-SAM - DUET`
  - `## 8) NPO-SAM - UNLamb`

## 2) Issues found and fixes applied

### 2.1 Hydra strict override compatibility

Two Hydra failures were found during smoke runs:

1. Missing-key failure on some model configs:
   - `model.model_args.low_cpu_mem_usage`
2. Missing-key failure on eval:
   - `model.model_args.base_model_name_or_path`

Initial fix used `+...` (append).  
Then 8B runs showed `+...` fails when key already exists.

Final robust fix:

- switched to `++...` (add-or-override) in both scripts:
  - `scripts/popqa/npo_sam_popqa.sh`
  - `scripts/duet/npo_sam_duet.sh`

Specifically:

- `++model.model_args.low_cpu_mem_usage=true` (train + eval)
- `++model.model_args.base_model_name_or_path=...` (eval)

This now works for both cases:

- configs where keys are absent (e.g., some 1B configs)
- configs where keys are already present (8B configs)

## 3) What was checked

### 3.1 Static checks

- `bash -n scripts/popqa/npo_sam_popqa.sh`
- `bash -n scripts/duet/npo_sam_duet.sh`

Both passed.

### 3.2 Runtime prerequisites and data/model accessibility

Verified:

- GPU is visible in host execution context.
- HF repos are accessible:
  - `SwetieePawsss/UNLamb_ft_models`
  - `SwetieePawsss/DUET_ft_models`
  - datasets `SwetieePawsss/exp_UNLamb` and `SwetieePawsss/DUET`
- Required splits resolve and load:
  - POPQA: `rare_forget5_sum`, `popular_forget5_sum`, `fast_retain_500`
  - DUET: `city_forget_rare_5`, `city_forget_popular_5`, `city_fast_retain_500`

### 3.3 Small-model pipeline smoke (1B)

Purpose:

- validate full train + eval pipeline quickly
- verify NPO-SAM logging and artifact generation

Runs used:

- `BASE_MODEL=Llama-3.2-1B-Instruct`
- `MODEL_CONFIG=Llama-3.2-1B-Instruct-lora`
- `LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models`
- `SFT_SUBFOLDER=llama-3.2-1b-instruct-popqa-ft`
- `MERGE_POPULARITY_FORGET=1`
- method args:
  - `BETAS=0.1`
  - `ALPHAS=1.0`
  - `GAMMAS=1.0`
  - `SAM_RHOS=0.01`
  - `SAM_ADAPTIVES=false`
  - `SAM_EPS=1e-12`
- training args:
  - `LRS=1e-5`
  - `PER_DEVICE_TRAIN_BS=1`
  - `GRAD_ACCUM=32`
  - `NUM_EPOCHS=0.03`
  - `LORA_RS=32`
  - `LORA_ALPHAS=64`
  - `LORA_DROPOUTS=0.0`

Result: POPQA 1B smoke run (train + eval) succeeded

- run dir:
  - `/workspace/unlearning/saves/unlearn/popqa/npo_sam/popqa_Llama-3.2-1B-Instruct_forget5_sum_npo_sam_lora_r32_lalpha64_ldrop0p0_lr1e-5_beta0p1_alpha1p0_gamma1p0_rho0p01_adF`
- summary:
  - `forget_qa_rouge`: `0.28651202749140897`
  - `holdout_qa_rouge`: `0.10333333333333333`
- trainer state:
  - `global_step=2`, `max_steps=2`, `train_runtime=52.2863s`

Result: DUET 1B smoke run (train + eval) succeeded

- run dir:
  - `/workspace/unlearning/saves/unlearn/duet/npo_sam/duet_Llama-3.2-1B-Instruct_city_forget_5_npo_sam_lora_r32_lalpha64_ldrop0p0_lr1e-5_beta0p1_alpha1p0_gamma1p0_rho0p01_adF`
- summary:
  - `forget_qa_rouge`: `0.2664851313969571`
  - `holdout_qa_rouge`: `0.3706666666666667`
- trainer state:
  - `global_step=1`, `max_steps=1`, `train_runtime=27.9585s`

### 3.4 Production-like smoke on referenced 8B checkpoints

Purpose:

- verify scripts in a setup matching `prod-runs.md`
- keep training minimal while preserving full train+eval flow

Runs used:

- `USE_SFT_BASE=1`
- `MERGE_POPULARITY_FORGET=1`
- same method/training args as above
- `NUM_EPOCHS=0.02` (minimal smoke; still performed optimizer step)

UNLamb 8B run:

- base:
  - `LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models`
  - `SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft`
- script:
  - `bash scripts/popqa/npo_sam_popqa.sh`
- run dir:
  - `/workspace/unlearning/saves/unlearn/popqa/npo_sam/popqa_Llama-3.1-8B-Instruct_forget5_sum_npo_sam_lora_r32_lalpha64_ldrop0p0_lr1e-5_beta0p1_alpha1p0_gamma1p0_rho0p01_adF`
- summary:
  - `forget_qa_rouge`: `0.820303550973654`
  - `holdout_qa_rouge`: `0.8168333333333333`
- trainer state:
  - `global_step=1`, `max_steps=1`, `train_runtime=35.8008s`

DUET 8B run:

- base:
  - `LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models`
  - `SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft`
- script:
  - `bash scripts/duet/npo_sam_duet.sh`
- run dir:
  - `/workspace/unlearning/saves/unlearn/duet/npo_sam/duet_Llama-3.1-8B-Instruct_city_forget_5_npo_sam_lora_r32_lalpha64_ldrop0p0_lr1e-5_beta0p1_alpha1p0_gamma1p0_rho0p01_adF`
- summary:
  - `forget_qa_rouge`: `0.9382434301521438`
  - `holdout_qa_rouge`: `0.9675`
- trainer state:
  - `global_step=1`, `max_steps=1`, `train_runtime=38.3406s`

## 4) Commands used for production-like 8B smoke checks

### 4.1 UNLamb (POPQA) 8B

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
LRS=1e-5 \
BETAS=0.1 \
ALPHAS=1.0 \
GAMMAS=1.0 \
SAM_RHOS=0.01 \
SAM_ADAPTIVES=false \
SAM_EPS=1e-12 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
NUM_EPOCHS=0.02 \
LORA_RS=32 \
LORA_ALPHAS=64 \
LORA_DROPOUTS=0.0 \
FORCE_RERUN=1 \
bash scripts/popqa/npo_sam_popqa.sh
```

### 4.2 DUET 8B

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
LRS=1e-5 \
BETAS=0.1 \
ALPHAS=1.0 \
GAMMAS=1.0 \
SAM_RHOS=0.01 \
SAM_ADAPTIVES=false \
SAM_EPS=1e-12 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
NUM_EPOCHS=0.02 \
LORA_RS=32 \
LORA_ALPHAS=64 \
LORA_DROPOUTS=0.0 \
FORCE_RERUN=1 \
bash scripts/duet/npo_sam_duet.sh
```

## 5) Full production command pattern

For full runs with default script hyperparameters (same style as other algorithms), use the commands added in:

- `prod-runs.md`
  - `## 7) NPO-SAM - DUET`
  - `## 8) NPO-SAM - UNLamb`

## 6) Notes

- In this environment, GPU visibility is available in host execution context.
- NPO-SAM logs were emitted during training (`npo_sam_forget_loss_1`, `npo_sam_forget_loss_2`, `npo_sam_retain_loss`, `npo_sam_grad_norm`, `npo_sam_rho`, `npo_sam_adaptive`), confirming trainer path execution.
