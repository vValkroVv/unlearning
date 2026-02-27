# FALCON Integration and Validation Notes

Date: 2026-02-26
Repo: `/workspace/unlearning`

## 1) What was implemented

### 1.1 FALCON trainer

Implemented a dedicated FALCON trainer:

- `src/trainer/unlearn/falcon.py`

Core behavior implemented:

- Forget objective (InfoNCE) with POV construction from SVD principal directions.
- Retain objective default aligned with paper Eq. (11): cosine alignment.
- Gradient conflict handling with orthogonal projection when cosine conflict is detected.
- LoRA frozen-reference forward path uses `disable_adapter()` with eval-mode toggling for deterministic frozen activations.

### 1.2 Trainer/config wiring

- Registered trainer in:
  - `src/trainer/__init__.py`
- Added trainer config:
  - `configs/trainer/FALCON.yaml`

### 1.3 Dedicated FALCON experiments and scripts

Added dedicated experiment configs:

- `configs/experiment/unlearn/duet/falcon_lora.yaml`
- `configs/experiment/unlearn/popqa/falcon_lora.yaml`

Added standalone run scripts:

- `scripts/duet/falcon_duet.sh`
- `scripts/popqa/falcon_popqa.sh`

### 1.4 FALCON defaults adjusted after validation

To make FALCON runs stable by default:

- `gradient_checkpointing: false` set in FALCON experiment configs:
  - `configs/experiment/unlearn/duet/falcon_lora.yaml`
  - `configs/experiment/unlearn/popqa/falcon_lora.yaml`
- FALCON scripts now default to checkpointing off, with env override support:
  - `GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}`
  - passed as `trainer.args.gradient_checkpointing=...`

### 1.5 Paper-style MI layer selection + sweep defaults

Implemented MI-guided layer selection utility:

- `src/tools/falcon_mi_select.py`

Estimator behavior:

- Per-layer MI over forget/retain activations.
- Multi-domain aggregate MI objective with `eta` weighting.
- PCA reduction before KDE (`pca_var` threshold).
- Gaussian KDE with Scott bandwidth.

Integrated optional MI preselection in run scripts:

- `scripts/duet/falcon_duet.sh`
- `scripts/popqa/falcon_popqa.sh`

Activation (off by default):

- `MI_SELECT_LAYERS=1`

Common MI env overrides:

- `MI_MODEL_CFG`, `MI_MODEL_PATH`, `MI_TOKENIZER_PATH`
- `MI_DATASET_PATH`, `MI_FORGET_SPLITS`, `MI_RETAIN_SPLIT`
- `MI_ETA`, `MI_PCA_VAR`, `MI_MAX_EXAMPLES`, `MI_TOPK`
- `MI_DEVICE`, `MI_BATCH_SIZE`, `MI_OUT_DIR`

Sweep defaults in both FALCON scripts now:

- `K_SVDS=2,4,8,16`
- `ALPHAS=1,2,4`
- `GAMMAS=1,2,4`

## 2) POPQA eval file naming fix

Question raised: why POPQA wrote `DUET_SUMMARY.json`.

Root cause:

- POPQA reused `DUETEvaluator` and hardcoded evaluator name `"DUET"`, so outputs were named `DUET_EVAL.json` / `DUET_SUMMARY.json`.

Fix implemented:

- `src/evals/duet.py` now reads evaluator name from config (`name`, default `"DUET"`).
- `configs/eval/popqa.yaml` now sets:
  - `name: POPQA`

Result:

- POPQA eval outputs are now:
  - `POPQA_EVAL.json`
  - `POPQA_SUMMARY.json`

POPQA scripts were updated accordingly (skip/rerun artifact paths):

- `scripts/popqa/falcon_popqa.sh`
- `scripts/popqa/ga_popqa.sh`
- `scripts/popqa/gd_popqa.sh`
- `scripts/popqa/npo_popqa.sh`
- `scripts/popqa/wga_popqa.sh`
- `scripts/popqa/ada_wgd_popqa.sh`
- `scripts/popqa/ada_pop_popqa.sh`

## 3) What was checked

### 3.1 Environment preflight

Checked in active `.venv`:

- GPU availability via `nvidia-smi`
- imports/versions for `torch`, `transformers`, `datasets`, `peft`

Observed:

- GPU visible and idle at test start.
- Core packages imported successfully.

### 3.2 Static checks

- `bash -n` for FALCON scripts.
- `python3 -m py_compile` for FALCON trainer/evaluator files.
- grep checks for expected wiring (`trainer=FALCON`, `retain_mode=cosine`, checkpointing default false).

All passed.

### 3.3 Data/model accessibility

Checked:

- `SwetieePawsss/exp_UNLamb` splits:
  - `rare_forget5_sum`
  - `fast_retain_500`
- tokenizer from `unsloth/Llama-3.1-8B-Instruct`

All loaded successfully.

### 3.4 FALCON smoke train/eval

Ran 1-step smoke training and eval on sliced POPQA splits with unsloth base.

Successful run artifacts:

- `/workspace/unlearning/saves/unlearn/popqa/falcon/smoke_default_gc_false_check/`
- Eval outputs:
  - `/workspace/unlearning/saves/unlearn/popqa/falcon/smoke_default_gc_false_check/evals_popqa_name/POPQA_EVAL.json`
  - `/workspace/unlearning/saves/unlearn/popqa/falcon/smoke_default_gc_false_check/evals_popqa_name/POPQA_SUMMARY.json`

Trainer logs confirmed FALCON metrics were produced (example keys):

- `falcon_forget_loss`
- `falcon_retain_loss`
- `falcon_grad_cos`
- `falcon_conflict`

## 4) Known runtime notes

- If using script defaults with `USE_SFT_BASE=0` and no model-path override, base resolves to gated Meta repo:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - requires HF auth/access.
- For open access, use:
  - `HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct`
  - `TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct`

## 5) Recommended launch pattern

```bash
FORCE_RERUN=1 \
USE_SFT_BASE=0 \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
NUM_EPOCHS=0.01 \
GRAD_ACCUM=1 \
PER_DEVICE_TRAIN_BS=1 \
LRS="1e-5" \
K_SVDS="2,4,8,16" \
ALPHAS="1,2,4" \
GAMMAS="1,2,4" \
MI_SELECT_LAYERS=1 \
MI_TOPK=1 \
RETAIN_MODES="cosine" \
bash scripts/popqa/falcon_popqa.sh
```
