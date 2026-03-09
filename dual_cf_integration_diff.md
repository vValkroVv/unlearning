# DualCF Integration Diff

Base commit: `3e15a8ba7682cf316469a6ffc417c62d33aa22b1` (before DualCF integration)
Target: current working tree

## What was added

DualCF is integrated as a new routed unlearning method that keeps the repo's
existing `forget.original` / `forget.alternate` DPO-style batch structure, but
optimizes the practical decomposition

\[
CE(y^{cf}\mid x) + \lambda_i^{neg} L_{NPO}(x, y^{orig}) + \alpha_{eff} L_{ret},
\]

with:

- per-sample difficulty routing on the forget side,
- per-sample attribution routing on the forget side,
- batch-level `alpha_eff` as the retain-side proxy that fits the current
  `GradDiff` trainer flow.

## Changed files

### Core trainer path

- `src/trainer/utils.py`
- `src/trainer/unlearn/dual_cf.py`
- `src/trainer/__init__.py`
- `configs/trainer/DualCF.yaml`

### Dataset / collator plumbing

- `src/data/qa.py`
- `src/data/collators.py`
- `src/data/__init__.py`
- `src/data/utils.py`
- `configs/data/datasets/DUET_QA_forget_dual_cf.yaml`
- `configs/data/datasets/POPQA_QA_forget_dual_cf.yaml`
- `configs/data/datasets/RWKU_QA_forget_dual_cf.yaml`

### Experiment configs

- `configs/experiment/unlearn/duet/dual_cf_lora.yaml`
- `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`

### Launch scripts

- `scripts/duet/dual_cf_duet.sh`
- `scripts/popqa/dual_cf_popqa.sh`
- `scripts/rwku/dual_cf_rwku.sh`

### Offline artifact tools

- `src/tools/dual_cf_artifact_utils.py`
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`

## Core implementation notes

### `src/trainer/utils.py`

Added per-sample helpers without changing legacy DPO/NPO helpers:

```python
def _token_counts_from_labels(labels: torch.Tensor) -> torch.Tensor:
    counts = (labels[..., 1:] != -100).sum(dim=-1)
    return counts.clamp_min(1)

def compute_nll_per_sample(model, inputs, normalize_by_tokens: bool = True):
    ...

def compute_npo_per_sample(model, ref_model, lose_inputs, beta: float = 1.0, ...):
    ...
```

This preserves routed per-sample signals and avoids length bias from raw sequence
sums.

### `src/trainer/unlearn/dual_cf.py`

New `DualCF(GradDiff)` trainer:

- uses `compute_nll_per_sample()` on `forget.alternate`,
- uses `compute_npo_per_sample()` on `forget.original`,
- computes soft routing gates
  - `s = sigmoid((difficulty - tau_d) / temp_d)`
  - `r = sigmoid((attribution - tau_a) / temp_a)`
- applies
  - `lambda_neg = lambda_neg_max * s * (1 - r)`
  - `forget_scale = 1 - (1 - risk_forget_scale) * r`
  - `alpha_eff = alpha * (lambda_ret_lo + (lambda_ret_hi - lambda_ret_lo) * r.mean())`
- logs `dualcf_*` diagnostics for smoke tests and sweeps.

Important repo-fit constraint:

- retain weighting is implemented as batch-level `alpha_eff`, not literal
  per-sample `lambda_i^{ret}`, because `GradDiff` consumes one scalar retain loss
  from a separate retain batch.

### Dataset / collator path

`QAwithAlternateMetadataDataset` returns:

```python
{
    "original": tokenized_original,
    "alternate": tokenized_alternate,
    "difficulty_score": float(...),
    "attribution_score": float(...),
    "index": int(...),
}
```

`DataCollatorForSupervisedDataset` now recursively stacks numeric scalar metadata,
so routed scores survive batching without a method-specific collator.

`add_dataset_index()` now no-ops if an artifact already contains an `index`
column, which is required for premerged DualCF datasets.

## Artifact schema

The expected forget-side artifact row is:

```json
{
  "index": 17,
  "question": "...",
  "answer": "...",
  "alternate": "...",
  "difficulty_score": 0.73,
  "attribution_score": 0.18
}
```

The experiment configs support either:

- a Hugging Face dataset path / repo in `cf_dataset_path`, or
- a local JSON/JSONL path by setting:
  - `cf_dataset_path=json`
  - `cf_dataset_data_files=/abs/path/to/file.jsonl`

## Launch configuration

Each benchmark now has a `dual_cf_lora.yaml` experiment config plus a launcher
script with env-controlled routing knobs:

- `BETAS`
- `TAU_DS`
- `TAU_AS`
- `TEMP_DS`
- `TEMP_AS`
- `LAMBDA_NEG_MAXS`
- `LAMBDA_RET_LOS`
- `LAMBDA_RET_HIS`
- `CF_WEIGHTS`
- `RISK_FORGET_SCALES`
- `DISABLE_DIFFICULTY_ROUTES`
- `DISABLE_ATTRIBUTION_ROUTES`

Operational update:

- experiment configs now default to local JSON artifact mode with
  - `cf_dataset_path=json`
  - `cf_dataset_split=train`
- launcher scripts now fail fast if:
  - `CF_DATASET_PATH` still uses a placeholder path
  - local JSON mode is selected without `CF_DATASET_DATA_FILES`
- launcher scripts now support controlled smoke-test execution with:
  - `MAX_STEPS`
  - `FORGET_SPLIT_OVERRIDE`
  - `RETAIN_SPLIT_OVERRIDE`
  - `FORGET_LABEL_OVERRIDE`
- DUET and POPQA switch back to benchmark-native split names only when
  `CF_DATASET_PATH` is overridden away from `json`
- RWKU local JSON mode defaults to `cf_dataset_name=null` and
  `cf_dataset_split=train`
- the GPU validation playbook now lives in `plan-test-dual.md`, including:
  - full-file JSONL integrity scanning via environment variables
  - a 1B DUET launcher smoke command with `USE_SFT_BASE=0`
  - an explicit train-then-eval DUET small-run procedure

This supports the intended ablations without additional trainer classes:

- full DualCF
- difficulty-only
- attribution-only

Uniform counterfactual remains available through the existing `DPO` trainer.

## Offline tooling

### `src/tools/make_counterfactuals.py`

Builds `alternate` answers by:

- copying an existing alternate column,
- joining from a JSONL sidecar,
- or generating one alternate answer per sample with a model config.

### `src/tools/score_difficulty.py`

Builds `difficulty_score` using cheap offline proxies:

- optional inverted MRD column,
- popularity normalization,
- model confidence normalization,
- optional stage prior.

### `src/tools/score_attribution.py`

Builds `attribution_score` from a proxy retain bank by:

- averaging retain gradients on trainable params,
- scoring forget gradients by dot-product or cosine alignment,
- clipping negatives to zero,
- min-max normalizing the positive risk signal.

## Smoke-test command

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  task_name=duet_dualcf_smoke \
  model=Llama-3.2-1B-Instruct-lora \
  "forget_split='city_forget_rare_5[:2]'" \
  "retain_split='city_fast_retain_500[:2]'" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

Expected smoke-test checks:

- `inputs["forget"]["original"]` exists
- `inputs["forget"]["alternate"]` exists
- `difficulty_score` and `attribution_score` reach `DualCF.compute_loss()`
- `dualcf_*` logs appear
- one adapter checkpoint is saved
