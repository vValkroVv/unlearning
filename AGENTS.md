# AGENTS.md

## Mission
This repository is for controlled / selective unlearning with DualCF and strong split-matched comparisons against DPO, GA, NPO, NPO-SAM, and LoKU.

Optimize for three things together:
1. valid offline artifacts,
2. fair ablations and stable campaigns,
3. thesis-grade analysis of forget, holdout/locality, utility, and trajectory stability.

## Read first before changing code
For most DualCF work, read these in order:
1. `README.md`
2. `dual_cf_v3_integration_diff.md`
3. `prod-run-dual-gpu-v3.md`
4. `unlearning.txt`
5. `docs/experiments.md`
6. `docs/repro.md`
7. the exact launcher or tool you are modifying

Use `dual_cf_v2_integration_diff.md` only when you need the frozen pre-v3 DualCF baseline snapshot.

## Repository map for DualCF work
Core trainer path:
- `src/trainer/unlearn/dual_cf.py`
- `src/trainer/utils.py`
- `configs/trainer/DualCF.yaml`
- `src/trainer/__init__.py`

Dataset / collator path:
- `src/data/qa.py`
- `src/data/collators.py`
- `configs/data/datasets/*dual_cf*.yaml`

Offline artifact path:
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/build_proxy_retain_map.py`
- `src/tools/score_attribution.py`
- `src/tools/calibrate_dual_cf_scores.py`
- `src/tools/validate_dual_cf_artifact.py`
- `src/tools/dual_cf_artifact_utils.py`

Campaign / launcher path:
- `scripts/duet/*.sh`
- `scripts/rwku/*.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `dual-scripts-run/*.sh`

Analysis path:
- `src/tools/summarize_checkpoint_metrics.py`
- `src/tools/summarize_utility_metrics.py`
- `structured-saves.zip` outputs
- `dualcf_detailed_report.md`

## Hard invariants
- Keep artifact preparation offline. Do not move attribution scoring into `DualCF.compute_loss()`.
- Preserve the `forget.original` / `forget.alternate` batch contract unless the task explicitly redesigns the trainer API.
- Preserve split matching. Rare, popular, merged, and RWKU artifacts must be prepared and trained separately.
- In local JSON mode, `cf_dataset_split` controls the actual train slice. `forget_split` still controls benchmark identity.
- Never compare a merged DualCF artifact against rare-only or popular-only baseline runs.
- Always validate artifacts before training.
- Keep provenance sidecars for final artifacts.
- Keep LoRA parity between offline attribution scoring and training unless the experiment explicitly studies mismatch.
- Always report both endpoint metrics and trajectory metrics.
- Utility and semantic forgetting matter. Do not optimize only `forget_qa_rouge`.

## Working style
- Treat each task like a GitHub issue. Be specific about files, commands, constraints, and the exact split or dataset.
- Prefer small functional runs before full H100 campaigns.
- When editing scripts, keep baseline parity across DualCF / DPO / GA / NPO / NPO-SAM / LoKU unless the task is intentionally method-specific.
- When changing run management, preserve checkpoint evaluation and summary generation.
- When changing artifact tooling, fail fast with clear row/index context.

## Validation ladder
When changing artifact tooling:
1. smoke on a tiny slice,
2. run `validate_dual_cf_artifact.py`,
3. inspect score ranges and invalid row counts,
4. run a 1-step train smoke,
5. run a short functional train + eval,
6. only then schedule the full campaign.

When changing trainer code:
1. keep old configs working,
2. add new config knobs explicitly,
3. verify `dualcf_*` logs,
4. run a tiny train smoke,
5. run a short DUET rare or RWKU short slice,
6. only then add to the main ablation tree.

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
- forget cosine discussion,
- trajectory stability summary,
- concrete next ablations.

## Codex agent map
Use only these three custom Codex agents:
- `spark-discovery-agent`
- `portable-runtime-engineer`
- `docs-sync-editor`

Official Codex structure for this repo:
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
If code or launchers changed, finish with `dualcf-doc-sync`.
