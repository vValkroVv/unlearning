# VAST Working Log: GPU venv setup for `unlearning`

Date: 2026-02-25
Repo: `/workspace/unlearning`

## 1) Read setup instructions and inspect dependency files

Commands run:

```bash
sed -n '1,260p' README.md
sed -n '1,260p' requirements.txt
sed -n '1,260p' setup.py
```

Findings:
- README suggests:
  - `pip install .[lm_eval]`
  - `pip install --no-build-isolation flash-attn==2.6.3`
- `setup.py` loads deps directly from `requirements.txt`.
- `requirements.txt` includes non-PyPI ROS packages (example: `actionlib==1.14.0`) that fail on this server.

## 2) Create virtual environment

Commands run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3) Try direct README install and record failure

Command run:

```bash
pip install '.[lm-eval]'
```

Result:
- Failed because of non-portable deps from `requirements.txt`:
  - `ERROR: No matching distribution found for actionlib==1.14.0`

## 4) Install working dependency set

### 4.1 Install core runtime deps used by codebase

```bash
pip install \
  numpy==2.2.3 hydra-core==1.3.0 hydra-colorlog==1.2.0 omegaconf==2.3.0 \
  transformers==4.45.1 accelerate==0.34.2 datasets==3.0.1 peft==0.15.2 \
  deepspeed==0.15.4 scipy==1.14.1 tqdm==4.67.1 rouge-score==0.1.2 \
  scikit-learn==1.5.2 huggingface-hub==0.29.1 sentencepiece==0.2.1 \
  evaluate==0.4.3 lm-eval==0.4.8 jsonlines==4.0.0 pytorch-revgrad==0.2.0 \
  einops==0.8.1 pandas==2.3.0
```

### 4.2 Install CUDA PyTorch (GPU build)

First, GPU availability was validated:

```bash
nvidia-smi
```

Then replaced CPU torch with CUDA torch:

```bash
pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Resolved a dependency conflict caused by torch reinstall:

```bash
pip install fsspec==2024.6.1
```

### 4.3 Install flash-attention

```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn==2.6.3
```

Build result:
- `flash-attn-2.6.3` built successfully from source and installed.

## 5) Configure local writable cache directories

To avoid permission/cache issues on this server session:

```bash
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
mkdir -p "$HF_HOME/hub" "$TRITON_CACHE_DIR"
```

## 6) Sanity checks

### 6.1 Dependency consistency

```bash
pip check
```

Result:
- `No broken requirements found.`

### 6.2 Full module import sweep (`src/`)

Command run:

```bash
python - <<'PY'
import os, sys, importlib
src='/workspace/unlearning/src'
sys.path.insert(0, src)
mods=[]
for root, _, files in os.walk(src):
    for f in files:
        if f.endswith('.py'):
            rel=os.path.relpath(os.path.join(root,f), src)
            m=rel[:-3].replace('/', '.')
            if m.endswith('.__init__'): m=m[:-9]
            if m: mods.append(m)
mods=sorted(set(mods))
fails=[]
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        fails.append((m, type(e).__name__, str(e)))
print('TOTAL_MODULES', len(mods))
print('FAILED', len(fails))
for m,et,msg in fails:
    print(f"{m}::{et}::{msg}")
PY
```

Result:
- `TOTAL_MODULES 50`
- `FAILED 0`

### 6.3 Entry point checks

```bash
python src/train.py --help
python src/eval.py --help
python setup_data.py --help
```

Result:
- All three commands executed successfully.

### 6.4 GPU + flash-attn runtime smoke test

```bash
python - <<'PY'
import torch
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_func
print('flash_attn_version', getattr(flash_attn,'__version__','unknown'))
print('torch', torch.__version__, 'cuda_build', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
q = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 16, 8, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
print('flash_attn_out_shape', tuple(out.shape))
print('flash_attn_out_dtype', out.dtype)
print('gpu_name', torch.cuda.get_device_name(0))
PY
```

Result:
- `flash_attn_version 2.6.3`
- `torch 2.4.1+cu124`, CUDA build `12.4`
- `cuda_available True`
- Output tensor computed on GPU (`NVIDIA RTX A6000`)

## 7) Ready-to-use activation commands

Run in each new shell:

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
mkdir -p "$HF_HOME/hub" "$TRITON_CACHE_DIR"
```

## Notes
- `setup.py` currently uses `find_packages()` but repo code lives under `src/` without package discovery metadata, so direct entry scripts (`python src/train.py ...`, `python src/eval.py ...`) are the reliable path.
- In restricted sandbox contexts, CUDA visibility checks can report false negatives. GPU checks above were validated in full GPU-visible execution context.
- The import sweep covers `src/tools/` as well as training code, so the environment also needs `openai` for `src/tools/vllm_cf_client.py` and `src/tools/make_counterfactuals.py`.

## 8) Extra packages required specifically for `scripts/duet/npo_duet.sh`

Reason:
- `configs/trainer/finetune.yaml` sets:
  - `optim: paged_adamw_32bit` (requires `bitsandbytes`)
  - `report_to: tensorboard` (requires `tensorboard`)

Install command used:

```bash
pip install bitsandbytes==0.44.1 tensorboard==2.20.0 wandb==0.21.0
```

Post-install checks:

```bash
pip check
python -m bitsandbytes
```

Result:
- `pip check` passed with no broken requirements.
- `python -m bitsandbytes` ended with `SUCCESS! Installation was successful!` in GPU-visible context.

## 9) 2026-03-14 verification fix

While rerunning `setup_vast_env.sh`, the scripted import sweep initially failed with:

```text
tools.make_counterfactuals::ModuleNotFoundError::No module named 'openai'
tools.vllm_cf_client::ModuleNotFoundError::No module named 'openai'
```

Fix applied:

```bash
pip install openai==1.109.1
```

The setup script was updated so future runs install `openai` during the main dependency step. After that change:

- `pip check` passed.
- Full `src/` import sweep passed (`TOTAL_MODULES 71`, `FAILED 0`).
- `python src/train.py --help`, `python src/eval.py --help`, and `python setup_data.py --help` passed.
- `python -m bitsandbytes` ended with `SUCCESS! Installation was successful!`.
- FlashAttention GPU smoke test passed on `NVIDIA RTX A5000` with `torch 2.4.1+cu124` and `flash-attn 2.6.3`.

Targeted optimizer/reporting smoke test:

```bash
python - <<'PY'
import torch
from transformers import Trainer, TrainingArguments

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8, 8)
    def forward(self, input_ids=None, labels=None):
        y = self.l(torch.randn(2,8, device=self.l.weight.device))
        return {'loss': y.mean(), 'logits': y}

args = TrainingArguments(
    output_dir='/tmp/bnb_smoke2',
    per_device_train_batch_size=1,
    report_to=['tensorboard'],
    optim='paged_adamw_32bit',
    max_steps=1,
)
model = Tiny().cuda() if torch.cuda.is_available() else Tiny()
trainer = Trainer(model=model, args=args)
trainer.create_optimizer()
print('optimizer_class', trainer.optimizer.__class__.__name__)
print('optimizer_module', trainer.optimizer.__class__.__module__)
print('device', next(model.parameters()).device)
PY
```

Result:
- `optimizer_module bitsandbytes.optim.adamw`
- Confirms `paged_adamw_32bit` path is active.

## 9) Runtime note for `npo_duet.sh`

Default local checkpoint in script:

```bash
/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb
```

Current server status:
- This path is missing.

To run script, use one of:

```bash
# Option A: use HF base model directly (requires HF access to meta-llama model)
USE_SFT_BASE=0 bash scripts/duet/npo_duet.sh

# Option B: point to a local checkpoint you have
LOCAL_SFT_BASE=/path/to/your/local/base/checkpoint bash scripts/duet/npo_duet.sh
```

## 10) `ga_duet.sh` readiness for `unsloth/Llama-3.1-8B-Instruct`

Objective:
- Prepare everything so `scripts/duet/ga_duet.sh` can run with unsloth Llama-3.1-8B-Instruct.
- Do not run `ga_duet.sh` directly; instead, validate all internal code paths with lightweight smoke commands.

### 10.1 Small script fix needed (tokenizer path override)

Issue found:
- `ga_duet.sh` already overrides model weights path (`model.model_args.pretrained_model_name_or_path`) but not tokenizer path.
- With unsloth weights, this left tokenizer loading on default `meta-llama/...` from model config, causing:
  - `401 Client Error: Unauthorized` for gated `meta-llama/Llama-3.1-8B-Instruct`.

Patch applied in `scripts/duet/ga_duet.sh`:
- Added:
  - `tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${hf_base_model_path}}"`
- Passed tokenizer override to both train and eval calls:
  - `model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}`

This keeps previous behavior by default, and enables clean unsloth usage via env override.

### 10.2 Preflight checks run

Command:

```bash
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_home/datasets
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"
nvidia-smi
```

Result:
- GPU visible: `NVIDIA RTX A6000` (CUDA-enabled context).

Command (dataset + tokenizer accessibility):

```bash
python - <<'PY'
from datasets import load_dataset
from transformers import AutoTokenizer
print('forget_len', len(load_dataset('SwetieePawsss/DUET', split='city_forget_rare_5')))
print('retain_len', len(load_dataset('SwetieePawsss/DUET', split='city_fast_retain_500')))
tok = AutoTokenizer.from_pretrained('unsloth/Llama-3.1-8B-Instruct')
print('tokenizer_ok', tok.__class__.__name__)
PY
```

Result:
- DUET splits loaded successfully:
  - `city_forget_rare_5`: `482`
  - `city_fast_retain_500`: `500`
- Tokenizer load successful from `unsloth/Llama-3.1-8B-Instruct`.

### 10.3 Lightweight train smoke (same code path as `ga_duet.sh`)

Command (1 step, tiny split slices, LoRA small rank for speed):

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/grad_ascent_lora.yaml \
  trainer=GradAscent \
  task_name=duet_smoke_unsloth_ga_lora \
  model=Llama-3.1-8B-Instruct-lora \
  "forget_split='city_forget_rare_5[:2]'" \
  "retain_split='city_fast_retain_500[:2]'" \
  model.model_args.pretrained_model_name_or_path=unsloth/Llama-3.1-8B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=unsloth/Llama-3.1-8B-Instruct \
  model.model_args.device_map=auto \
  model.model_args.low_cpu_mem_usage=true \
  model.lora_config.r=8 \
  model.lora_config.lora_alpha=16 \
  model.lora_config.lora_dropout=0.0 \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  trainer.args.logging_steps=1 \
  retain_logs_path=null \
  paths.output_dir=/workspace/unlearning/saves/unlearn/duet/ga/duet_smoke_unsloth_ga_lora
```

Result:
- Train path completed successfully.
- Artifacts created:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - tokenizer files and trainer state in run dir.

### 10.4 Lightweight eval smoke (same code path as `ga_duet.sh`)

Command:

```bash
python src/eval.py \
  experiment=eval/duet/default.yaml \
  model=Llama-3.1-8B-Instruct-lora \
  "forget_split='city_forget_rare_5[:2]'" \
  "holdout_split='city_fast_retain_500[:2]'" \
  task_name=duet_smoke_unsloth_ga_lora \
  model.model_args.pretrained_model_name_or_path=/workspace/unlearning/saves/unlearn/duet/ga/duet_smoke_unsloth_ga_lora \
  model.model_args.base_model_name_or_path=unsloth/Llama-3.1-8B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=unsloth/Llama-3.1-8B-Instruct \
  model.model_args.device_map=auto \
  model.model_args.low_cpu_mem_usage=true \
  model.lora_config.r=8 \
  model.lora_config.lora_alpha=16 \
  model.lora_config.lora_dropout=0.0 \
  eval.duet.overwrite=true \
  eval.duet.batch_size=1 \
  eval.duet.metrics.forget_qa_rouge.generation_args.max_new_tokens=8 \
  eval.duet.metrics.holdout_qa_rouge.generation_args.max_new_tokens=8 \
  paths.output_dir=/workspace/unlearning/saves/unlearn/duet/ga/duet_smoke_unsloth_ga_lora/evals \
  retain_logs_path=null
```

Result:
- Eval path completed successfully.
- Outputs:
  - `DUET_EVAL.json`
  - `DUET_SUMMARY.json` with keys `forget_qa_rouge`, `holdout_qa_rouge`.

### 10.5 `ga_duet.sh` logic checks

Checked:
- `bash -n scripts/duet/ga_duet.sh` -> syntax OK.
- `_splits.sh` behavior:
  - no merge: rare and popular runs separately
  - `MERGE_POPULARITY_FORGET=1`: merged forget split label works.
- Variable resolution for unsloth run:
  - `USE_SFT_BASE=0` makes base model path come from `HF_BASE_MODEL_PATH`.
  - `TOKENIZER_MODEL_PATH` now controls tokenizer source for both train and eval.

### 10.6 Final command to run full GA DUET with unsloth

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_home/datasets
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
MERGE_POPULARITY_FORGET=1 \
bash scripts/duet/ga_duet.sh
```

Optional fast sanity run before full sweep:

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
LRS="1e-5" \
LORA_RS="8" \
LORA_ALPHAS="16" \
LORA_DROPOUTS="0.0" \
NUM_EPOCHS=1 \
GRAD_ACCUM=1 \
PER_DEVICE_TRAIN_BS=1 \
MERGE_POPULARITY_FORGET=1 \
bash scripts/duet/ga_duet.sh
```

### 10.7 Exact command started (verbatim)

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_home/datasets
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
MERGE_POPULARITY_FORGET=1 \
bash scripts/duet/ga_duet.sh
```

## 11) Full `ga_duet.sh` run results (unsloth Llama-3.1-8B-Instruct)

Run context:
- Script: `scripts/duet/ga_duet.sh`
- Merge mode: `MERGE_POPULARITY_FORGET=1`
- Base model source: `unsloth/Llama-3.1-8B-Instruct`
- LoRA grid:
  - `LORA_RS=32`
  - `LORA_ALPHAS=64`
  - `LORA_DROPOUTS=0.0`
- Learning-rate sweep (default): `1e-5 5e-5 1e-4 5e-4 1e-3`

Outcome:
- Full sweep completed: `5/5` runs finished.
- All run directories contain expected train + eval artifacts:
  - train: `adapter_model.safetensors`, `adapter_config.json`, tokenizer/config files, trainer state/logs
  - eval: `evals/DUET_EVAL.json`, `evals/DUET_SUMMARY.json`, `evals/eval.log`

Result summaries (`DUET_SUMMARY.json`):

| LR   | forget_qa_rouge | holdout_qa_rouge |
|------|------------------|------------------|
| 1e-5 | 0.7045643153526971 | 0.8003333333333332 |
| 5e-5 | 0.5232451590594743 | 0.6377333333333334 |
| 1e-4 | 0.0 | 0.001 |
| 5e-4 | 0.0 | 0.0 |
| 1e-3 | 0.0 | 0.0 |

Notes:
- Best holdout retention in this sweep was at `lr=1e-5`.
- High learning rates (`>=1e-4`) collapsed both forget and holdout ROUGE toward zero in this setup.

## 12) `npo_duet.sh` start (unsloth Llama-3.1-8B-Instruct)

### 12.1 Small script fix needed (tokenizer path override)

Issue observed:
- `scripts/duet/npo_duet.sh` used `HF_BASE_MODEL_PATH` for model weights but did not override tokenizer source.
- This caused tokenizer fallback to gated `meta-llama/Llama-3.1-8B-Instruct` and a `401` access error.

Patch applied in `scripts/duet/npo_duet.sh`:
- Added:
  - `tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${hf_base_model_path}}"`
- Passed tokenizer override to both train and eval calls:
  - `model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}`

### 12.2 Exact command started (verbatim)

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_home/datasets
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
MERGE_POPULARITY_FORGET=1 \
bash scripts/duet/npo_duet.sh
```

## 13) Full `npo_duet.sh` run results (unsloth Llama-3.1-8B-Instruct)

Run context:
- Script: `scripts/duet/npo_duet.sh`
- Merge mode: `MERGE_POPULARITY_FORGET=1`
- Base model source: `unsloth/Llama-3.1-8B-Instruct`
- LoRA grid:
  - `LORA_RS=32`
  - `LORA_ALPHAS=64`
  - `LORA_DROPOUTS=0.0`
- NPO method args:
  - `BETAS=0.5`
  - `ALPHAS=1.0`
  - `GAMMAS=1.0`
- Learning-rate sweep (default): `1e-5 5e-5 1e-4 5e-4 1e-3`

Outcome:
- Full sweep completed: `5/5` runs finished.
- No active NPO train/eval process remained after completion check.
- All run directories contain expected train + eval artifacts:
  - train: `adapter_model.safetensors`, `adapter_config.json`, tokenizer/config files, trainer state/logs
  - eval: `evals/DUET_EVAL.json`, `evals/DUET_SUMMARY.json`, `evals/eval.log`

Result summaries (`DUET_SUMMARY.json`):

| LR   | forget_qa_rouge | holdout_qa_rouge |
|------|------------------|------------------|
| 1e-5 | 0.7006224066390041 | 0.7973333333333332 |
| 5e-5 | 0.6979598893499308 | 0.853 |
| 1e-4 | 0.4785788381742739 | 0.8015 |
| 5e-4 | 0.4073997233748271 | 0.8795 |
| 1e-3 | 0.07717842323651451 | 0.9581666666666666 |

Notes:
- Highest holdout retention in this sweep was at `lr=1e-3`.
- Forget-score reduction improved with higher learning rate in this run set (lowest forget ROUGE at `lr=1e-3`).

## 14) Run `falcon_duet.sh` with DUET dataset (unsloth Llama-3.1-8B-Instruct)

### 14.1 Runtime notes for FALCON DUET

- Script: `scripts/duet/falcon_duet.sh`
- DUET splits are sourced from `scripts/duet/_splits.sh`:
  - `city_forget_rare_5 city_fast_retain_500`
  - `city_forget_popular_5 city_fast_retain_500`
- FALCON defaults currently set for stable launch:
  - `RETAIN_MODES=cosine` by default in script
  - `GRADIENT_CHECKPOINTING=false` by default in script + FALCON experiment yaml
- If you use `USE_SFT_BASE=0` without overrides, script defaults to gated
  `meta-llama/Llama-3.1-8B-Instruct`; use unsloth paths to avoid auth failures.

### 14.2 Exact command to run FALCON on DUET (verbatim)

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_home/datasets
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"

CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
MERGE_POPULARITY_FORGET=1 \
bash scripts/duet/falcon_duet.sh
```

### 14.3 Optional fast sanity run before full sweep

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=unsloth/Llama-3.1-8B-Instruct \
MERGE_POPULARITY_FORGET=1 \
FORCE_RERUN=1 \
NUM_EPOCHS=0.01 \
GRAD_ACCUM=1 \
PER_DEVICE_TRAIN_BS=1 \
LRS="1e-5" \
K_SVDS="4" \
TARGET_LAYERS="7" \
RETAIN_MODES="cosine" \
bash scripts/duet/falcon_duet.sh
```

### 14.4 Expected artifacts

For each task under:

```bash
/workspace/unlearning/saves/unlearn/duet/falcon/<task_name>/
```

Expected train artifacts:

- `adapter_model.safetensors`
- `adapter_config.json`
- `trainer_state.json`

Expected eval artifacts:

- `evals/DUET_EVAL.json`
- `evals/DUET_SUMMARY.json`
