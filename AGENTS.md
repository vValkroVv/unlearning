# AGENTS.md

## Mission
This branch is for controlled / selective unlearning with DualCF v2 and split-matched comparisons on DUET, POPQA, and RWKU.

Optimize for three things together:
1. valid offline artifacts,
2. fair split-matched ablations and stable campaigns,
3. thesis-grade analysis of forget, holdout/locality, utility, and checkpoint trajectories.

## Read first before changing code
For most DualCF v2 work, read these in order:
1. `README.md`
2. `dual_cf_integration_diff.md`
3. `prod-run-dual-gpu.md`
4. `prod-run-dual-vast.md`
5. `unlearning.txt`
6. `docs/experiments.md`
7. `docs/repro.md`
8. the exact launcher, experiment config, or tool you are modifying

Use `prod-runs.md` and `prod-gpu-runs*.md` only when you need older generic launcher history outside the main DualCF v2 flow.

## Repository map for DualCF v2 work
Core trainer path:
- `src/trainer/unlearn/dual_cf.py`
- `src/trainer/utils.py`
- `configs/trainer/DualCF.yaml`
- `src/trainer/__init__.py`

Dataset / collator path:
- `src/data/qa.py`
- `src/data/collators.py`
- `src/data/utils.py`
- `src/data/__init__.py`
- `configs/data/datasets/*dual_cf*.yaml`

Experiment config path:
- `configs/experiment/unlearn/duet/dual_cf_lora.yaml`
- `configs/experiment/unlearn/duet/dual_cf_v2_lora.yaml`
- `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_v2_lora.yaml`

Offline artifact path:
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`
- `src/tools/build_proxy_retain_map.py`
- `src/tools/calibrate_dual_cf_scores.py`
- `src/tools/validate_dual_cf_artifact.py`
- `src/tools/dual_cf_artifact_utils.py`

Campaign / launcher path:
- `scripts/duet/*.sh`
- `scripts/popqa/*.sh`
- `scripts/rwku/*.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `prod-run-dual-gpu.md`
- `prod-run-dual-vast.md`

Analysis / eval path:
- `src/tools/build_utility_1k_panel.py`
- `src/tools/summarize_checkpoint_metrics.py`
- `src/tools/summarize_utility_metrics.py`
- `artifacts/evals/utility_1k_v1/*`
- `artifacts/duet/*_v2/*.jsonl`
- `artifacts/rwku/*_v2/*.jsonl`

## Hard invariants
- Keep counterfactual generation, difficulty scoring, attribution scoring, and calibration offline. Do not move them into `DualCF.compute_loss()`.
- Preserve the `forget.original` / `forget.alternate` batch contract unless the task explicitly redesigns the trainer API.
- Preserve routed metadata in artifacts and batching: `index`, `difficulty_score`, and `attribution_score` must survive dataset loading and collation.
- Preserve split matching. DUET `rare`, `popular`, and `merged` artifacts are separate training objects; do not compare merged DualCF runs against rare-only or popular-only baselines.
- In local JSON mode, `cf_dataset_split` controls the actual train slice. `forget_split` remains the benchmark identity.
- Keep LoRA parity between offline scoring passes and training unless the experiment explicitly studies mismatch.
- Always validate artifacts before training.
- Keep Utility-1K panel reuse and checkpoint evaluation intact when editing run management.
- Always report both endpoint metrics and trajectory metrics.
- Do not optimize only `forget_qa_rouge`; utility and holdout/locality still matter.

## Working style
- Treat each task like a GitHub issue. Be specific about files, commands, constraints, and the exact dataset or forget label.
- Prefer tiny functional runs before full H100 or VAST campaigns.
- When editing scripts, keep baseline parity across DualCF and the branch baselines (`GA`, `GD`, `NPO`, `NPO-SAM`, `LoKU`, `FALCON`, `R2D`, `AdaPop`, `AdaWGD`, `WGA`) unless the task is intentionally method-specific.
- When changing run management, preserve `OUTPUT_ROOT`, half-epoch checkpoint evaluation, Utility-1K evaluation, and cleanup behavior.
- When changing artifact tooling, fail fast with clear row / index context.
- When touching DUET production flows, preserve the main `rare -> popular -> merged` ordering unless the task explicitly changes campaign orchestration.

## Validation ladder
When changing artifact tooling:
1. smoke on a tiny slice,
2. run `validate_dual_cf_artifact.py`,
3. inspect score ranges, calibration outputs, and invalid row counts,
4. run a 1-step train smoke,
5. run a short DUET `rare` or RWKU functional train + eval,
6. only then schedule the wider `rare -> popular -> merged` or RWKU campaign.

When changing trainer code:
1. keep existing configs working, including `dual_cf_lora.yaml` and `dual_cf_v2_lora.yaml`,
2. add new config knobs explicitly,
3. verify `dualcf_*` logs plus retain / routing diagnostics,
4. run a tiny train smoke,
5. run a short DUET `rare` or RWKU slice with checkpoint eval,
6. only then add the change to the main ablation tree.

When changing launchers or runbooks:
1. preserve offline artifact inputs, `OUTPUT_ROOT`, and eval cadence,
2. smoke one method on one split with one LR,
3. verify endpoint eval plus Utility-1K summary generation,
4. only then scale to the full sweep.

## What good outputs look like
For code tasks, deliver:
- exact files changed,
- why the change is safe,
- exact commands to test,
- what metrics or logs should move,
- any fairness risks or confounds.

For analysis tasks, deliver:
- matched-forget comparison,
- utility breakdown by task,
- holdout/locality summary,
- routing / calibration summary,
- trajectory stability summary,
- concrete next ablations.

## Codex agent map
Use only these three custom Codex agents:
- `spark-discovery-agent`
- `portable-runtime-engineer`
- `docs-sync-editor`

Official Codex structure for this branch:
- custom subagents live in `.codex/agents/*.toml`
- shared agent settings live in `.codex/config.toml`
- skills live in `.agents/skills/*/SKILL.md`

Model selection belongs in `.codex/agents/*.toml` or `.codex/config.toml`, not in this `AGENTS.md`.

## Skill map
Use these repo-local skills when relevant:
- `dualcf-fast-discovery`
- `dualcf-portable-runtime`
- `dualcf-doc-sync`

If a task starts unclear, run `dualcf-fast-discovery` first.
If code or launcher files changed, finish with `dualcf-doc-sync`.
