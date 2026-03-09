# DualCF Test Log

Date: 2026-03-09
Repo: `/workspace/unlearning`

This file is the running verification log for DualCF. It records what was
actually tested, what failed, what was fixed, and what remains.

## 1. Scope

Goal for this pass:

1. verify the new DualCF pipeline end to end on controlled DUET and RWKU slices
2. fix concrete integration failures instead of only reviewing code
3. keep a reproducible log of commands, artifacts, and outcomes

## 2. Environment preflight

Verified against the venv setup documented in `vast-working.md`.

Commands run:

```bash
source .venv/bin/activate
python -V
pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Observed:

1. Python `3.10.12`
2. `pip check` reported `No broken requirements found.`
3. GPU visible: `NVIDIA RTX A6000, 49140 MiB`

## 3. Repo and integration inspection

Read and cross-checked:

1. `dual_cf_integration_diff.md`
2. `plan-test-dual.md`
3. `npo_sam_integration_diff.md`
4. `scripts/duet/dual_cf_duet.sh`
5. `scripts/rwku/dual_cf_rwku.sh`
6. `src/trainer/unlearn/dual_cf.py`
7. `src/trainer/utils.py`
8. `src/data/qa.py`
9. `src/data/collators.py`
10. `src/tools/make_counterfactuals.py`
11. `src/tools/score_difficulty.py`
12. `src/tools/score_attribution.py`

Verified from code review:

1. DualCF trainer is registered and wired through configs.
2. Metadata fields `difficulty_score` and `attribution_score` are expected by
   the trainer and collator path.
3. DUET/RWKU experiment configs use local JSON mode by default.
4. Existing `dual_cf` launch scripts already enforce missing-artifact failures.

## 4. HF reachability and dataset shape checks

Used the user-provided HF token through environment variables during checks.

Repos confirmed reachable:

1. model repo `SwetieePawsss/DUET_ft_models`
2. dataset repo `SwetieePawsss/DUET`
3. dataset repo `SwetieePawsss/exp_r`

Split/schema checks run:

```bash
python - <<'PY'
import datasets
checks = [
 ('SwetieePawsss/DUET', None, 'city_forget_rare_5'),
 ('SwetieePawsss/DUET', None, 'city_forget_popular_5'),
 ('SwetieePawsss/DUET', None, 'city_fast_retain_500'),
 ('SwetieePawsss/DUET', None, 'city_forget_rare_5+city_forget_popular_5'),
 ('SwetieePawsss/exp_r', 'forget_level2', 'test'),
 ('SwetieePawsss/exp_r', 'neighbor_level2', 'test'),
]
for path, name, split in checks:
    ds = datasets.load_dataset(path, name=name, split=split)
    print(path, name, split, len(ds), ds.column_names)
PY
```

Observed sizes:

1. `city_forget_rare_5`: `482`
2. `city_forget_popular_5`: `482`
3. `city_fast_retain_500`: `500`
4. `city_forget_rare_5+city_forget_popular_5`: `964`
5. `forget_level2/test`: `2879`
6. `neighbor_level2/test`: `5533`

Verified key columns:

1. DUET has `question`, `answer`, `pop_sum`
2. RWKU has `query`, `answer`, `pop_sum`

## 5. First concrete failure found

### 5.1 Failure

The first run of `make_counterfactuals.py` on a DUET rare slice failed when
loading `meta-llama/Llama-3.2-1B-Instruct`.

Command that failed:

```bash
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5[:4]' \
  --output-path /tmp/duet_rare_step1_small.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --max-new-tokens 32
```

Observed failure:

1. gated repo access error while fetching model weights
2. shared model loader did not explicitly pass the active HF token into
   `transformers.from_pretrained(...)`

### 5.2 Fix applied

Patched the shared loaders to forward the active HF token from environment
variables:

1. `src/model/__init__.py`
2. `src/model/lora.py`
3. `src/data/utils.py`
4. `src/tools/dual_cf_artifact_utils.py`

Behavior after fix:

1. model loads succeed when `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set
2. gated model access now works in artifact tools and training paths

## 6. DUET small-slice artifact pipeline

Ran the full three-stage artifact pipeline on a controlled rare DUET slice to
verify plumbing before scaling to full artifacts.

### 6.1 Counterfactual generation

Command:

```bash
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5[:4]' \
  --output-path /tmp/duet_rare_step1_small.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --max-new-tokens 32
```

Result:

1. succeeded
2. output file created: `/tmp/duet_rare_step1_small.jsonl`
3. runtime about `1m13s` for 4 examples

Note:

1. transformers emitted a Flash Attention warning because the inference model
   is initialized before being moved to GPU
2. this was a warning only; the command completed successfully

### 6.2 Difficulty scoring

Command:

```bash
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files /tmp/duet_rare_step1_small.jsonl \
  --output-path /tmp/duet_rare_step2_small.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --batch-size 2
```

Result:

1. succeeded
2. output file created: `/tmp/duet_rare_step2_small.jsonl`
3. runtime about `14.5s`

### 6.3 Attribution scoring

Command:

```bash
python src/tools/score_attribution.py \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files /tmp/duet_rare_step2_small.jsonl \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500[:8] \
  --output-path /tmp/duet_rare_dualcf_small.jsonl \
  --question-key question \
  --retain-max-steps 8
```

Result:

1. succeeded
2. output file created: `/tmp/duet_rare_dualcf_small.jsonl`
3. runtime about `2m`
4. attribution stage is currently the slowest verified part of the pipeline

## 7. DUET artifact validation on the small slice

Quick preview confirmed rows contain:

1. `index`
2. `question`
3. `answer`
4. `alternate`
5. `difficulty_score`
6. `attribution_score`

Validation script run:

```bash
python - <<'PY'
import json, math
from pathlib import Path
path = Path('/tmp/duet_rare_dualcf_small.jsonl')
required = {'index','question','answer','alternate','difficulty_score','attribution_score'}
seen = set(); dups = set(); bad=[]
mins = {'difficulty_score': float('inf'), 'attribution_score': float('inf')}
maxs = {'difficulty_score': float('-inf'), 'attribution_score': float('-inf')}
for line_no, line in enumerate(path.open(), start=1):
    row = json.loads(line)
    missing = sorted(required - set(row))
    if missing: bad.append((line_no, f'missing {missing}'))
    idx = row.get('index')
    if idx in seen: dups.add(idx)
    seen.add(idx)
    for key in ('question','answer','alternate'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            bad.append((line_no, f'bad {key}'))
    for key in ('difficulty_score','attribution_score'):
        val = row.get(key)
        if not isinstance(val, (int,float)) or not math.isfinite(val):
            bad.append((line_no, f'bad {key}: {val!r}'))
        else:
            mins[key] = min(mins[key], float(val)); maxs[key] = max(maxs[key], float(val))
    if row.get('answer','').strip().lower() == row.get('alternate','').strip().lower():
        bad.append((line_no, 'answer matches alternate'))
print('rows', len(seen))
print('dups', sorted(dups))
print('bad', bad)
print('ranges', {k:(mins[k], maxs[k]) for k in mins})
PY
```

Observed:

1. `rows 4`
2. `dups []`
3. `bad []`
4. `difficulty_score` range: `(0.0, 0.9686862134725694)`
5. `attribution_score` range: `(0.0, 1.0)`

## 8. DUET direct 1-step smoke

### 8.1 First attempt after the interruption

State verified first:

1. `/tmp/duet_dualcf_smoke` did not exist
2. no train result from the interrupted attempt was reused

First command retried:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_smoke \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files=/tmp/duet_rare_dualcf_small.jsonl \
  cf_dataset_split=train[:2] \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

Observed failure:

1. Hydra parse error: `mismatched input '[' expecting <EOF>`
2. cause: bracket slicing must be passed as a quoted Hydra string

### 8.2 Fix for local JSON slice syntax

Verified working forms include:

1. `\"cf_dataset_split='train[:2]'\"`
2. launcher-array form `"cf_dataset_split='${cf_dataset_split}'"`

Code fix applied to launcher scripts:

1. `scripts/duet/dual_cf_duet.sh`
2. `scripts/popqa/dual_cf_popqa.sh`
3. `scripts/rwku/dual_cf_rwku.sh`

### 8.3 Second attempt

After fixing quoting, the next direct smoke exposed another issue:

1. the run still tried to load `meta-llama/Llama-3.1-8B-Instruct`
2. cause: the `dual_cf_lora.yaml` experiment configs hardcoded 8B model paths,
   defeating `model=Llama-3.2-1B-Instruct-lora`

Code fix applied:

1. removed hardcoded base-model paths from:
   - `configs/experiment/unlearn/duet/dual_cf_lora.yaml`
   - `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
   - `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`
2. result: the selected `/model` config now controls the base model unless the
   launcher explicitly overrides it

### 8.4 Final direct smoke result

Final working command:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_smoke \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files=/tmp/duet_rare_dualcf_small.jsonl \
  "cf_dataset_split='train[:2]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

Observed result:

1. succeeded
2. runtime about `34.8s`
3. `dualcf_cf_loss` logged
4. `dualcf_neg_loss` logged
5. `dualcf_forget_loss` logged
6. `dualcf_alpha_eff` logged
7. no `KeyError` for `difficulty_score`
8. no `KeyError` for `attribution_score`
9. no `_score_tensor` batch-size mismatch
10. output directory created: `/tmp/duet_dualcf_smoke`
11. adapter files created:
   - `adapter_model.safetensors`
   - `adapter_config.json`

## 9. DUET launcher smoke

Launcher command run:

```bash
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/tmp/duet_rare_dualcf_small.jsonl \
CF_DATASET_SPLIT='train[:2]' \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.2-1B-Instruct \
HF_BASE_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct \
MODEL_CONFIG=Llama-3.2-1B-Instruct-lora \
TOKENIZER_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct \
NUM_EPOCHS=1 \
MAX_STEPS=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=1 \
LRS=1e-5 \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_HIS=2.0 \
RISK_FORGET_SCALES=0.5 \
FORCE_RERUN=1 \
FORGET_SPLIT_OVERRIDE=city_forget_rare_5 \
RETAIN_SPLIT_OVERRIDE=city_fast_retain_500 \
FORGET_LABEL_OVERRIDE=city_forget_rare_5_smoke \
bash scripts/duet/dual_cf_duet.sh
```

Observed result:

1. succeeded
2. training launched with the HF 1B base
3. `MAX_STEPS=1` was honored
4. evaluation launched and completed
5. run directory created under:
   `/workspace/unlearning/saves/unlearn/duet/dual_cf/duet_Llama-3.2-1B-Instruct_city_forget_rare_5_smoke_dual_cf_lora_r32_lalpha64_ldrop0p0_lr1e-5_beta0p5_alpha1p0_gamma1p0_td0p5_ta0p5_sd0p25_sa0p25_ln1p0_rlo1p0_rhi2p0_cf1p0_rf0p5_dOn_aOn`
6. expected eval outputs exist:
   - `evals/DUET_EVAL.json`
   - `evals/DUET_SUMMARY.json`

Smoke metrics observed:

1. `forget_qa_rouge = 0.13131051175656985`
2. `holdout_qa_rouge = 0.4753333333333334`

## 10. DUET small functional run

### 10.1 Larger rare-only artifact for the functional run

Built a 32-example rare-only artifact:

```bash
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5[:32]' \
  --output-path /tmp/duet_rare_step1_32.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --max-new-tokens 32
```

```bash
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files /tmp/duet_rare_step1_32.jsonl \
  --output-path /tmp/duet_rare_step2_32.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --batch-size 4
```

Initial attribution attempt on the full 1B base model was too slow for the
functional run. Switched to LoRA-only attribution scoring.

### 10.2 Tool fix for offline LoRA attribution

New failure:

1. `score_attribution.py` with the 1B LoRA config failed with a CPU/GPU device
   mismatch
2. cause: `load_model_bundle()` inherited `device_map=auto` from training-time
   configs, but the artifact tools expect to control device placement

Code fix applied:

1. `src/tools/dual_cf_artifact_utils.py`
2. behavior change: offline artifact tools now clear `model_args.device_map`
   before loading the model

### 10.3 Final 32-example attribution run

Working command:

```bash
python src/tools/score_attribution.py \
  --model-cfg configs/model/Llama-3.2-1B-Instruct-lora.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files /tmp/duet_rare_step2_32.jsonl \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500[:16] \
  --output-path /tmp/duet_rare_dualcf_32.jsonl \
  --question-key question \
  --retain-max-steps 16 \
  --lora-only
```

Result:

1. succeeded
2. runtime about `2m40s`
3. validation passed:
   - `rows 32`
   - `dups []`
   - `bad_count 0`
4. score ranges:
   - `difficulty_score`: `(0.186515278188893, 0.921045152534558)`
   - `attribution_score`: `(0.0, 1.0)`

### 10.4 Direct small train run

Command:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_small \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files=/tmp/duet_rare_dualcf_32.jsonl \
  "cf_dataset_split='train[:32]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.num_train_epochs=1 \
  trainer.args.learning_rate=1e-5 \
  trainer.args.logging_steps=1 \
  paths.output_dir=/tmp/duet_dualcf_small
```

Result:

1. succeeded
2. runtime about `1m6s`
3. finite `dualcf_*` logs appeared throughout training
4. no OOM
5. output written to `/tmp/duet_dualcf_small`

### 10.5 Direct small eval run

First eval attempt failed because sliced Hydra split overrides also need quoting:

1. `forget_split=city_forget_rare_5[:32]` failed to parse
2. `holdout_split=city_fast_retain_500[:64]` failed to parse

Second eval attempt failed because `base_model_name_or_path` is not declared in
the 1B LoRA config and must be appended with `++`.

Final working command:

```bash
python src/eval.py \
  experiment=eval/duet/default.yaml \
  model=Llama-3.2-1B-Instruct-lora \
  "forget_split='city_forget_rare_5[:32]'" \
  "holdout_split='city_fast_retain_500[:64]'" \
  task_name=duet_dualcf_small \
  model.model_args.pretrained_model_name_or_path=/tmp/duet_dualcf_small \
  ++model.model_args.base_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  model.model_args.device_map=auto \
  ++model.model_args.low_cpu_mem_usage=true \
  paths.output_dir=/tmp/duet_dualcf_small/evals \
  retain_logs_path=null
```

Result:

1. succeeded
2. runtime about `54s`
3. `forget_qa_rouge = 0.03125`
4. `holdout_qa_rouge = 0.453125`
5. eval outputs written to `/tmp/duet_dualcf_small/evals`

## 11. RWKU artifact build and smoke

### 11.1 First RWKU artifact attempt exposed a generator bug

Initial RWKU artifact build used the current `make_counterfactuals.py`
generation path and then failed validation:

1. `rows 16`
2. `dups []`
3. `bad_count 3`
4. failing rows had `alternate == answer`

Root cause:

1. the generation path asked the model the original QA question
2. it did not explicitly instruct the model to produce a counterfactual answer

Code fix applied:

1. `src/tools/make_counterfactuals.py`
2. generation mode now:
   - includes the true answer in the prompt
   - asks explicitly for a short incorrect alternative answer
   - retries with a stricter counterfactual wording if the first output still
     matches the true answer

### 11.2 Rebuilt RWKU artifact after the generator fix

Counterfactual generation:

```bash
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/exp_r \
  --dataset-name forget_level2 \
  --split 'test[:16]' \
  --output-path /tmp/rwku_step1_16_cf.jsonl \
  --question-key query \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --max-new-tokens 32
```

Validation of stage-1 alternates:

1. `matches 0` where `alternate == answer`

Difficulty scoring:

```bash
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files /tmp/rwku_step1_16_cf.jsonl \
  --output-path /tmp/rwku_step2_16_cf.jsonl \
  --question-key query \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --batch-size 4
```

Attribution scoring:

```bash
python src/tools/score_attribution.py \
  --model-cfg configs/model/Llama-3.2-1B-Instruct-lora.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files /tmp/rwku_step2_16_cf.jsonl \
  --retain-dataset-path SwetieePawsss/exp_r \
  --retain-dataset-name neighbor_level2 \
  --retain-split 'test[:16]' \
  --output-path /tmp/rwku_dualcf_16_cf.jsonl \
  --question-key query \
  --retain-question-key query \
  --retain-max-steps 16 \
  --lora-only
```

Final RWKU artifact validation:

1. `rows 16`
2. `dups []`
3. `bad_count 0`
4. `difficulty_score` range: `(0.08404723964584057, 0.9279552182665578)`
5. `attribution_score` range: `(0.0, 1.0)`

### 11.3 RWKU 1-step smoke

Command:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/rwku/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=rwku_dualcf_smoke \
  forget_split=forget_level2 \
  retain_split=neighbor_level2 \
  cf_dataset_path=json \
  cf_dataset_data_files=/tmp/rwku_dualcf_16_cf.jsonl \
  cf_dataset_name=null \
  "cf_dataset_split='train[:2]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/rwku_dualcf_smoke
```

Result:

1. succeeded
2. runtime about `27.4s`
3. `question_key=query` path worked correctly
4. `dualcf_*` logs appeared during training
5. output directory created: `/tmp/rwku_dualcf_smoke`
6. adapter files created:
   - `adapter_model.safetensors`
   - `adapter_config.json`

## 12. Added helper for future full artifacts

Added:

1. `src/tools/validate_dual_cf_artifact.py`

Purpose:

1. validate schema before training
2. fail on duplicate indices
3. fail on empty/non-string question-answer fields
4. fail on non-finite or non-numeric score fields
5. fail on `alternate == answer`
6. print score ranges for quick inspection

Verified:

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /tmp/duet_rare_dualcf_32.jsonl \
  --question-key question
```

```bash
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path /tmp/rwku_dualcf_16_cf.jsonl \
  --question-key query
```

Both commands succeeded.

## 13. Current status

Completed:

1. environment preflight
2. repo/config inspection
3. HF reachability checks
4. DUET schema checks
5. auth bug fix in shared loaders
6. DUET small-slice artifact generation
7. DUET small-slice artifact validation
8. DUET direct 1-step smoke
9. DUET launcher smoke
10. DUET small functional train/eval
11. RWKU artifact build
12. RWKU 1-step smoke
13. reusable artifact validator

Pending:

1. full DUET artifact preparation for rare/popular/merged
2. update `dual_cf_integration_diff.md` with verification-driven changes
