---
name: dualcf-doc-sync
description: Use after code, config, or launcher changes to keep integration diffs, runbooks, AGENTS guidance, and reproduce commands synchronized with the current repo state.
---

# DualCF Doc Sync

## When to use
- Any code or launcher change landed and docs are now stale.
- A runbook or integration diff must reflect a newly validated flow.
- AGENTS guidance or repo-local skill guidance should change with the code.
- You need concise but precise commands for local GPU and VAST users.

## Read first
- `AGENTS.md`
- `git diff`
- the relevant `*_integration_diff.md`
- `prod-run-dual-gpu.md`
- `prod-run-dual-vast.md`
- any README or docs page affected by the change

## Workflow
1. Inspect the actual code diff first.
2. Update the closest documentation source, not a redundant copy only.
3. Record exact files changed, exact commands, and exact validation status.
4. Keep local GPU and VAST instructions aligned where the flow is logically the same.
5. State residual gaps explicitly instead of implying full validation.

## Guardrails
- Never claim a run or test was completed if it was not.
- Keep commands copy-pasteable.
- Preserve repo terminology: split matching, offline artifacts, checkpoint eval, utility eval, trajectory metrics.
- Update integration diffs when behavior changed, not only prose docs.

## Output standard
Always return:
- exact docs changed,
- exact commands added or updated,
- validation status,
- caveats or follow-up docs still needed.
