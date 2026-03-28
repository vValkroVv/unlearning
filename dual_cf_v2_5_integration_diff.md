# DualCF v2.5 Integration Diff

Base branch: `dualfc_v2`  
Target: current working tree on `dualfc_v2_5`

## Scope

This branch adds three new `METHOD_VARIANTS` as first-class integrations:

- `multicf`
- `boundary_cf`
- `span_cf`

They are integrated in the same style already used by `simple_ce`:

- separate trainer implementations,
- separate trainer configs,
- separate DUET / RWKU experiment configs,
- wrapper dispatch by method name,
- shared DUET / RWKU launcher reuse,
- campaign-level per-method artifact routing,
- result/save parser support for new method names.

The branch changes 35 files total:

- 15 modified existing files
- 20 new files

## Exact File Coverage

### Modified existing files

#### `src/trainer/utils.py`

Added reusable token-weighted loss helpers and refactored the old per-sample NLL path so the new methods can share the same internals:

- `_compute_shifted_token_ce(...)`
- `_align_token_weights(...)`
- `compute_weighted_nll_per_sample(...)`
- `compute_weighted_npo_per_sample(...)`

The old `compute_nll_per_sample(...)` / `compute_npo_per_sample(...)` behavior remains available for existing methods.

#### `src/trainer/unlearn/dual_cf.py`

Refactored `DualCF` into subclass-friendly stages while keeping old routed DualCF behavior:

- `LOG_PREFIX`
- `_compute_cf_term(...)`
- `_compute_neg_term(...)`
- `_compute_routing(...)`
- `_forget_term_vector(...)`
- `_compute_core_components(...)`
- `_build_log_payload(...)`
- `_maybe_log(...)`
- `_extra_log_components(...)`

This is the base surface reused by `MultiCF`, `BoundaryCF`, and `SpanCF`.

#### `src/trainer/__init__.py`

Registered the new trainers:

- `MultiCF`
- `BoundaryCF`
- `SpanCF`

#### `src/data/qa.py`

Added two new dataset handlers:

- `QAMultiCFDataset`
- `QABoundaryCFDataset`

`QAMultiCFDataset` returns:

- `original`
- `alternates`
- `alternate_mask`
- `alternate_weights`
- requested numeric metadata

`QABoundaryCFDataset` extends the existing alternate+metadata pattern and additionally returns:

- `local_retain`
- `local_retain_index`

#### `src/data/collators.py`

Added `DataCollatorForMultiCF`.

This collator:

- pads a variable number of alternates across the batch,
- emits `alternates` as a list of per-alt batched tensors,
- emits `alternate_mask` as `[B, K]`,
- emits `alternate_weights` as `[B, K]`,
- preserves the existing recursive scalar metadata handling for the other fields.

#### `src/data/__init__.py`

Registered:

- `QAMultiCFDataset`
- `QABoundaryCFDataset`
- `DataCollatorForMultiCF`

#### `scripts/duet/run_dualcf_ablation_v2.sh`

Added explicit wrapper cases:

- `multicf -> MultiCF`
- `boundary_cf -> BoundaryCF`
- `span_cf -> SpanCF`

Each case sets:

- `TRAINER`
- `METHOD_NAME`
- `RUN_LABEL`
- `EXPERIMENT`

and then reuses `dual_cf_duet.sh`.

#### `scripts/rwku/run_dualcf_ablation_v2.sh`

Same integration pattern as DUET:

- `multicf -> MultiCF`
- `boundary_cf -> BoundaryCF`
- `span_cf -> SpanCF`

#### `scripts/duet/dual_cf_duet.sh`

Extended the shared DUET launcher so it can run:

- `DualCF`
- `MultiCF`
- `BoundaryCF`
- `SpanCF`

Key changes:

- method-specific env knobs were added:
  - `MULTICF_MAX_ALTERNATES_USED`
  - `MULTICF_ALT_AGG_MODE`
  - `MULTICF_ALT_WEIGHT_MODE`
  - `MULTICF_ALT_SET_TEMPERATURE`
  - `BOUNDARY_LOCAL_RETAIN_WEIGHT`
  - `BOUNDARY_MARGIN_WEIGHT`
  - `SPAN_MODE`
  - `SPAN_SHARED_TOKEN_WEIGHT`
  - `SPAN_UNIQUE_TOKEN_WEIGHT`
- method-specific suffixes are appended to `task_name`
- routed DualCF args are passed to all four routed trainers
- trainer-specific Hydra overrides are appended only for the relevant trainer

#### `scripts/rwku/dual_cf_rwku.sh`

Mirror of the DUET launcher changes for RWKU:

- same new env knobs
- same method-specific suffix logic
- same routed/common override handling

#### `scripts/dualcf/run_campaign_one_lr.sh`

This is the most important shell-level change.

Added:

- `resolve_duet_artifact_for_method(forget_label, method_variant)`
- `resolve_rwku_artifact_for_method(method_variant)`

Changed behavior:

- `CF_DATASET_DATA_FILES` is now resolved inside the per-method loop
- `multicf` uses `multicf_*_v1.jsonl`
- `boundary_cf` uses `boundarycf_*_v1.jsonl`
- `span_cf` keeps using the existing `dualcf_*_v2.jsonl`
- default `METHOD_VARIANTS` now includes:
  - `multicf`
  - `boundary_cf`
  - `span_cf`

Operational implication:

- the default campaign wrapper now expects the sibling `multicf_*` and `boundarycf_*` artifacts to exist if those methods are left in `METHOD_VARIANTS`

#### `check_saves.py`

Updated the default expected method list and matcher map to include:

- `multicf`
- `boundary_cf`
- `span_cf`

#### `src/tools/build_structured_saves.py`

Updated:

- `METHOD_RE`
- `METHOD_ORDER`

so structured-saves parsing recognizes the new method names and orders them deterministically.

#### `src/tools/build_results_combine_tables.py`

Updated:

- `WRONG_GENERATION_METHOD_MAP`
- `COMBINED_ROW_SPECS`

so combined tables can display:

- `MultiCF`
- `BoundaryCF`
- `SpanCF`

#### `src/tools/export_unlearning_sanity_checks.py`

Updated:

- `METHOD_RE`
- `METHOD_DISPLAY`
- `METHOD_ORDER`

so the sanity-report exporter no longer drops the new methods.

### New files

#### `src/trainer/unlearn/multicf.py`

New `MultiCF(DualCF)` trainer.

What it changes relative to `DualCF`:

- replaces only the counterfactual term
- consumes multiple alternates from `forget.alternates`
- aggregates them with `alternate_mask` and `alternate_weights`

Logged metrics:

- `multicf_num_alts_mean`
- `multicf_weight_entropy`
- `multicf_top1_share`

#### `src/trainer/unlearn/boundary_cf.py`

New `BoundaryCF(DualCF)` trainer.

What it changes relative to `DualCF`:

- keeps routed forget term
- keeps global retain term
- adds `local_retain_weight * local_retain_loss`
- optionally scales forget-side loss by `boundary_score`
- reads either `boundary_relation` / `boundary_overlap` or the older
  `boundary_relation_score` / `boundary_lexical_overlap` keys for logging

Logged metrics include:

- `boundary_score_mean`
- `boundary_margin_factor_mean`
- `boundary_relation_mean`
- `boundary_overlap_mean`
- `boundary_local_retain_loss`

#### `src/trainer/unlearn/span_cf.py`

New `SpanCF(DualCF)` trainer.

What it changes relative to `DualCF`:

- uses token-weighted NLL on `forget.alternate`
- uses token-weighted NPO on `forget.original`
- token weights are computed from gold/alternate overlap

Supported modes:

- `lcs`
- `set_overlap`

Logged metrics include:

- `span_alt_shared_token_frac`
- `span_alt_unique_token_frac`
- `span_orig_shared_token_frac`
- `span_orig_unique_token_frac`

#### `configs/trainer/MultiCF.yaml`

New trainer config for `MultiCF`.

Defaults added:

- `max_alternates_used: 4`
- `alt_agg_mode: weighted_mean`
- `alt_weight_mode: rerank`
- `alt_set_temperature: 0.7`

#### `configs/trainer/BoundaryCF.yaml`

New trainer config for `BoundaryCF`.

Defaults added:

- `local_retain_weight: 0.5`
- `boundary_margin_weight: 1.0`

#### `configs/trainer/SpanCF.yaml`

New trainer config for `SpanCF`.

Defaults added:

- `span_mode: lcs`
- `shared_token_weight: 0.25`
- `unique_token_weight: 1.0`

#### `configs/collator/DataCollatorForMultiCF.yaml`

New collator config selecting `DataCollatorForMultiCF`.

#### `configs/data/datasets/DUET_QA_forget_multicf.yaml`

New DUET forget-dataset config for `multicf`.

Uses:

- `QAMultiCFDataset`
- `alternate`
- `alternate_set`
- `alternate_set_weights`

#### `configs/data/datasets/RWKU_QA_forget_multicf.yaml`

RWKU equivalent of the DUET `multicf` dataset config.

#### `configs/data/datasets/DUET_QA_forget_boundary_cf.yaml`

New DUET forget-dataset config for `boundary_cf`.

Uses:

- `QABoundaryCFDataset`
- base routed metadata:
  - `difficulty_score`
  - `attribution_score`
- boundary metadata:
  - `boundary_score`
  - `boundary_relation`
  - `boundary_relation_score`
  - `boundary_shared_fact_score`
  - `boundary_type_match`
  - `boundary_overlap`
  - `boundary_lexical_overlap`

#### `configs/data/datasets/RWKU_QA_forget_boundary_cf.yaml`

RWKU equivalent of the DUET `boundary_cf` dataset config.

#### `configs/experiment/unlearn/duet/multicf_lora.yaml`

New DUET experiment config for `MultiCF`.

Overrides:

- `/trainer: MultiCF`
- `/data/datasets@data.forget: DUET_QA_forget_multicf`
- `/collator: DataCollatorForMultiCF`

#### `configs/experiment/unlearn/rwku/multicf_lora.yaml`

RWKU equivalent of the DUET `multicf` experiment config.

#### `configs/experiment/unlearn/duet/boundary_cf_lora.yaml`

New DUET experiment config for `BoundaryCF`.

Overrides:

- `/trainer: BoundaryCF`
- `/data/datasets@data.forget: DUET_QA_forget_boundary_cf`

#### `configs/experiment/unlearn/rwku/boundary_cf_lora.yaml`

RWKU equivalent of the DUET `boundary_cf` experiment config.

#### `configs/experiment/unlearn/duet/span_cf_lora.yaml`

New DUET experiment config for `SpanCF`.

Overrides:

- `/trainer: SpanCF`
- `/data/datasets@data.forget: DUET_QA_forget_dual_cf`

This method intentionally reuses the current DualCF artifact contract.

#### `configs/experiment/unlearn/rwku/span_cf_lora.yaml`

RWKU equivalent of the DUET `span_cf` experiment config.

#### `src/tools/build_multicf_artifact.py`

New offline artifact builder for `multicf`.

What it does:

- reads an existing artifact
- prefers the real external candidate family already present in `dualcf_new`
  (`external_alternates`, aligned score arrays, and keyed `api_sidecar.jsonl`)
- falls back to legacy sources only if the external pool is empty or unusable
- optionally merges in a candidate bank for that fallback path
- filters invalid candidates, then ranks valid ones by:
  - higher external/raw score
  - higher relation score
  - higher shared-fact score
  - lower `source_rank`
- rewrites the single `alternate` field to the top-ranked selected MultiCF
  alternate
- writes:
  - `alternate_set`
  - `alternate_set_weights`
  - `alternate_set_meta`
  - `multicf_source_pool`

Actual CLI surface currently added:

- `--input-path`
- `--output-path`
- `--candidate-bank`
- `--mapping-key`
- `--alternate-key`
- `--alternate-set-key`
- `--alternate-weights-key`
- `--candidate-field`
- `--external-alternates-key`
- `--external-scores-key`
- `--external-relation-scores-key`
- `--external-shared-fact-scores-key`
- `--external-sources-key`
- `--external-sidecar-path`
- `--top-k`
- `--reject-gold-substring`
- `--require-short-answer`
- `--max-overlap-ratio`
- `--max-alt-length-chars`
- `--sidecar-path`

#### `src/tools/build_boundary_cf_artifact.py`

New offline artifact builder for `boundary_cf`.

What it does:

- reads an existing artifact
- reads a proxy-retain map
- reads the retain dataset
- prefers the real external candidate family already present in `dualcf_new`
  (`external_alternates`, aligned score arrays, and keyed `api_sidecar.jsonl`)
- falls back to legacy sources only if the external pool is empty or unusable
- optionally reads a candidate bank for that fallback path
- scores candidates with the real relation/shared-fact metadata when present
- uses a staged selector:
  - `strict_overlap`
  - `relation_type_fallback`
  - `relation_only_fallback`
  - `valid_fallback`
- injects:
  - `boundary_score`
  - `boundary_relation`
  - `boundary_relation_score`
  - `boundary_shared_fact_score`
  - `boundary_type_match`
  - `boundary_overlap`
  - `boundary_lexical_overlap`
  - `boundary_selection_mode`
  - `boundary_source_pool`
  - `boundary_source`
  - `boundary_source_rank`
  - `local_retain_question`
  - `local_retain_answer`
  - `local_retain_index`

Actual CLI surface currently added:

- `--input-path`
- `--output-path`
- `--proxy-map-path`
- `--retain-dataset-path`
- `--retain-split`
- `--retain-dataset-name`
- `--retain-data-files`
- `--retain-question-key`
- `--retain-answer-key`
- `--candidate-bank`
- `--mapping-key`
- `--candidate-field`
- `--question-key`
- `--alternate-key`
- `--external-alternates-key`
- `--external-scores-key`
- `--external-relation-scores-key`
- `--external-shared-fact-scores-key`
- `--external-sources-key`
- `--external-sidecar-path`
- `--min-overlap-ratio`
- `--max-overlap-ratio`
- `--min-relation-score`
- `--reject-gold-substring`
- `--require-short-answer`
- `--max-alt-length-chars`
- `--sidecar-path`

#### `dual_cf_v2_5_integration_diff.md`

This document.

## Behavioral Summary

### `multicf`

Data path:

- sibling `multicf_*_v1.jsonl`
- `QAMultiCFDataset`
- `DataCollatorForMultiCF`
- `MultiCF`

Runtime path:

- method variant name: `multicf`
- trainer name: `MultiCF`
- DUET experiment: `unlearn/duet/multicf_lora.yaml`
- RWKU experiment: `unlearn/rwku/multicf_lora.yaml`

### `boundary_cf`

Data path:

- sibling `boundarycf_*_v1.jsonl`
- `QABoundaryCFDataset`
- `BoundaryCF`

Runtime path:

- method variant name: `boundary_cf`
- trainer name: `BoundaryCF`
- DUET experiment: `unlearn/duet/boundary_cf_lora.yaml`
- RWKU experiment: `unlearn/rwku/boundary_cf_lora.yaml`

### `span_cf`

Data path:

- existing `dualcf_*_v2.jsonl`
- existing `QAwithAlternateMetadataDataset`
- `SpanCF`

Runtime path:

- method variant name: `span_cf`
- trainer name: `SpanCF`
- DUET experiment: `unlearn/duet/span_cf_lora.yaml`
- RWKU experiment: `unlearn/rwku/span_cf_lora.yaml`

## Exact Commands Added / Now Supported

### Campaign wrapper

```bash
SEEDS="42 179 1137" METHOD_VARIANTS="multicf boundary_cf span_cf" \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

```bash
SEEDS="42 179 1137" METHOD_VARIANTS="multicf" \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

```bash
SEEDS="42 179 1137" METHOD_VARIANTS="boundary_cf" \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

```bash
SEEDS="42 179 1137" METHOD_VARIANTS="span_cf" \
bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all
```

### Direct wrapper dispatch

```bash
METHOD_VARIANT=multicf FORGET_LABEL=rare \
CF_DATASET_DATA_FILES=/abs/path/to/multicf_rare_v1.jsonl \
bash scripts/duet/run_dualcf_ablation_v2.sh
```

```bash
METHOD_VARIANT=boundary_cf \
CF_DATASET_DATA_FILES=/abs/path/to/boundarycf_forget_level2_v1.jsonl \
bash scripts/rwku/run_dualcf_ablation_v2.sh
```

```bash
METHOD_VARIANT=span_cf FORGET_LABEL=merged \
CF_DATASET_DATA_FILES=/abs/path/to/dualcf_merged_v2.jsonl \
bash scripts/duet/run_dualcf_ablation_v2.sh
```

### Artifact building

```bash
python src/tools/build_multicf_artifact.py \
  --input-path /abs/path/to/dualcf_rare_v2.jsonl \
  --candidate-bank /abs/path/to/rare_candidate_bank.jsonl \
  --output-path /abs/path/to/multicf_rare_v1.jsonl \
  --top-k 8 \
  --reject-gold-substring \
  --require-short-answer
```

```bash
python src/tools/build_boundary_cf_artifact.py \
  --input-path /abs/path/to/dualcf_rare_v2.jsonl \
  --proxy-map-path /abs/path/to/proxy_retain_map.jsonl \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500 \
  --candidate-bank /abs/path/to/rare_candidate_bank.jsonl \
  --output-path /abs/path/to/boundarycf_rare_v1.jsonl \
  --reject-gold-substring \
  --require-short-answer
```

## Validation Actually Completed

Completed in this turn:

- `python -m py_compile` on all changed Python modules
- `bash -n` on:
  - `scripts/dualcf/run_campaign_one_lr.sh`
  - `scripts/duet/run_dualcf_ablation_v2.sh`
  - `scripts/rwku/run_dualcf_ablation_v2.sh`
  - `scripts/duet/dual_cf_duet.sh`
  - `scripts/rwku/dual_cf_rwku.sh`
- YAML load check on all new config files
- `python src/tools/build_multicf_artifact.py --help`
- `python src/tools/build_boundary_cf_artifact.py --help`
- real `multicf_*` builds into:
  - `dualcf_new/duet/rare_llama31_8b_v2/multicf_rare_v1.jsonl`
  - `dualcf_new/duet/popular_llama31_8b_v2/multicf_popular_v1.jsonl`
  - `dualcf_new/duet/merged_llama31_8b_v2/multicf_merged_v1.jsonl`
  - `dualcf_new/rwku/llama31_8b_level2_v2/multicf_forget_level2_v1.jsonl`
- real `boundarycf_*` builds into:
  - `dualcf_new/duet/rare_llama31_8b_v2/boundarycf_rare_v1.jsonl`
  - `dualcf_new/duet/popular_llama31_8b_v2/boundarycf_popular_v1.jsonl`
  - `dualcf_new/duet/merged_llama31_8b_v2/boundarycf_merged_v1.jsonl`
  - `dualcf_new/rwku/llama31_8b_level2_v2/boundarycf_forget_level2_v1.jsonl`
- `validate_dual_cf_artifact.py` on all 8 generated artifacts with:
  - `--reject-gold-substring`
  - `--require-short-answer`
  - `--check-overlap-ratio 0.85`
- row-level sanity inspection of:
  - MultiCF alternate-count distributions
  - BoundaryCF selection-mode distributions
  - boundary relation / overlap / source-pool distributions

Observed build state on `dualcf_new`:

- `multicf` now uses the external candidate pool for every row across DUET rare,
  DUET popular, DUET merged, and RWKU level2.
- after the ranking fix, `alternate_set[0]` is always the max-weight selected
  candidate on all rebuilt MultiCF artifacts.
- MultiCF alternate-count means after real build:
  - DUET rare: `4.905`
  - DUET popular: `4.820`
  - DUET merged: `4.862`
  - RWKU level2: `5.627`
- `boundary_cf` also uses the external candidate pool for every row across those
  same splits.
- Boundary selection-mode mix after real build:
  - DUET rare: `strict_overlap=263`, `relation_type_fallback=201`,
    `relation_only_fallback=15`, `valid_fallback=3`
  - DUET popular: `strict_overlap=1`, `relation_type_fallback=440`,
    `relation_only_fallback=41`
  - DUET merged: `strict_overlap=264`, `relation_type_fallback=641`,
    `relation_only_fallback=56`, `valid_fallback=3`
  - RWKU level2: `strict_overlap=572`, `relation_type_fallback=1921`,
    `relation_only_fallback=305`, `valid_fallback=81`
- All 8 generated artifacts passed `validate_dual_cf_artifact.py` with
  `bad_rows_count=0`.

Not completed in this turn:

- no train smoke
- no DUET / RWKU endpoint eval
- no checkpoint eval
- no Utility-1K / Utility-3K eval

## Known Risks / Follow-up

### Artifact availability

Because `scripts/dualcf/run_campaign_one_lr.sh` now includes the new methods in
its default `METHOD_VARIANTS`, full campaign runs will require:

- `multicf_*_v1.jsonl`
- `boundarycf_*_v1.jsonl`

to exist at the expected sibling paths under `${ARTIFACT_ROOT}`.

### Boundary artifact quality

`build_boundary_cf_artifact.py` no longer hardcodes relation to `1.0`; it now
uses the external relation/shared-fact metadata from `dualcf_new` when present.

The real caveat after the verified build is different:

- the uploaded `dualcf_new` candidate pool often does not contain a lexical
  near-miss, especially on DUET popular
- because of that, many rows land in `relation_type_fallback` rather than
  `strict_overlap`

So BoundaryCF is now integrated, buildable, and validated structurally on the
current artifacts, but the semantic quality of the boundary negatives still
needs empirical review before large training campaigns.

### No GPU validation yet

The branch is wired and syntax-checked, but the validation ladder from
`AGENTS.md` still needs the next steps:

1. tiny artifact build
2. `validate_dual_cf_artifact.py`
3. 1-step train smoke
4. short DUET / RWKU functional train + eval
5. only then campaign-scale runs
