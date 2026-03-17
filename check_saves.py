#!/usr/bin/env python3
"""Check DualCF campaign saves created by run_campaign_one_lr.sh."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_LRS = ["5e-6", "1e-5", "5e-5", "1e-4"]
DEFAULT_VARIANTS = ["full", "d_only", "a_only", "dpo", "ga", "npo", "npo_sam", "loku"]

PHASE_PREFIXES = {
    "duet_rare": "duet_Llama-3.1-8B-Instruct_city_forget_rare_5_",
    "duet_popular": "duet_Llama-3.1-8B-Instruct_city_forget_popular_5_",
    "duet_merged": "duet_Llama-3.1-8B-Instruct_city_forget_5_",
    "rwku": "rwku_Llama-3.1-8B-Instruct_forget_level2_",
}

LEGACY_RWKU_LOKU_PREFIX = "rwku_Llama-3.1-8B-Instruct_merged_loku_"


def _is_dualcf_full(name: str) -> bool:
    return "_dual_cf_" in name and "_dOn_aOn" in name


def _is_dualcf_d_only(name: str) -> bool:
    return "_dual_cf_" in name and "_dOn_aOff" in name


def _is_dualcf_a_only(name: str) -> bool:
    return "_dual_cf_" in name and "_dOff_aOn" in name


def _contains(token: str) -> Callable[[str], bool]:
    return lambda name: token in name


VARIANT_MATCHERS: dict[str, Callable[[str], bool]] = {
    "full": _is_dualcf_full,
    "d_only": _is_dualcf_d_only,
    "a_only": _is_dualcf_a_only,
    "dpo": _contains("_dpo_cf_"),
    "ga": _contains("_ga_"),
    "npo": lambda name: "_npo_" in name and "_npo_sam_" not in name,
    "npo_sam": _contains("_npo_sam_"),
    "loku": _contains("_loku_"),
}


@dataclass(frozen=True)
class ExpectedRun:
    phase: str
    lr: str
    variant: str


@dataclass
class RunCheckResult:
    run_dir: Path
    issues: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path_to_saves", required=True, help="Path to saves/ or saves/unlearn")
    parser.add_argument(
        "--lrs",
        nargs="+",
        default=DEFAULT_LRS,
        help=f"Expected LR tags. Default: {' '.join(DEFAULT_LRS)}",
    )
    parser.add_argument(
        "--method_variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"Expected method variants. Default: {' '.join(DEFAULT_VARIANTS)}",
    )
    return parser.parse_args()


def resolve_unlearn_root(path_to_saves: str) -> Path:
    root = Path(path_to_saves).expanduser().resolve()
    if (root / "unlearn").is_dir():
        return root / "unlearn"
    return root


def list_run_dirs(unlearn_root: Path) -> list[Path]:
    if not unlearn_root.exists():
        raise FileNotFoundError(f"Save root does not exist: {unlearn_root}")
    return sorted(path for path in unlearn_root.iterdir() if path.is_dir())


def build_expected_runs(lrs: list[str], variants: list[str]) -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for phase in PHASE_PREFIXES:
        for lr in lrs:
            for variant in variants:
                runs.append(ExpectedRun(phase=phase, lr=lr, variant=variant))
    return runs


def find_matches(run_dirs: list[Path], expected: ExpectedRun) -> list[Path]:
    prefixes = [PHASE_PREFIXES[expected.phase]]
    if expected.phase == "rwku" and expected.variant == "loku":
        prefixes.append(LEGACY_RWKU_LOKU_PREFIX)
    variant_matcher = VARIANT_MATCHERS[expected.variant]
    lr_pattern = re.compile(rf"_lr{re.escape(expected.lr)}(?:_|$)")
    return [
        run_dir
        for run_dir in run_dirs
        if any(run_dir.name.startswith(prefix) for prefix in prefixes)
        and lr_pattern.search(run_dir.name) is not None
        and variant_matcher(run_dir.name)
    ]


def require_file(path: Path, label: str, issues: list[str]) -> None:
    if not path.is_file():
        issues.append(label)


def require_duet_eval_artifacts(eval_dir: Path, run_dir: Path, issues: list[str]) -> None:
    relative = eval_dir.relative_to(run_dir)
    require_file(eval_dir / "DUET_EVAL.json", f"{relative}/DUET_EVAL.json", issues)
    require_file(eval_dir / "DUET_SUMMARY.json", f"{relative}/DUET_SUMMARY.json", issues)
    require_file(eval_dir / "COS_SIM_EVAL.json", f"{relative}/COS_SIM_EVAL.json", issues)
    require_file(eval_dir / "COS_SIM_SUMMARY.json", f"{relative}/COS_SIM_SUMMARY.json", issues)


def inspect_run_dir(run_dir: Path) -> RunCheckResult:
    issues: list[str] = []

    require_duet_eval_artifacts(run_dir / "evals", run_dir, issues)

    require_file(run_dir / "checkpoint_evals" / "summary.tsv", "checkpoint_evals/summary.tsv", issues)
    require_file(
        run_dir / "checkpoint_evals_utility" / "summary.tsv",
        "checkpoint_evals_utility/summary.tsv",
        issues,
    )
    require_file(
        run_dir / "checkpoint_evals_merged" / "summary.tsv",
        "checkpoint_evals_merged/summary.tsv",
        issues,
    )
    require_file(
        run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json",
        "checkpoint_evals_merged/trajectory_metrics.json",
        issues,
    )

    checkpoint_eval_root = run_dir / "checkpoint_evals"
    checkpoint_dirs = sorted(
        path
        for path in checkpoint_eval_root.glob("checkpoint-*")
        if path.is_dir()
    )
    if not checkpoint_dirs:
        issues.append("checkpoint_evals/checkpoint-*")
    else:
        for checkpoint_dir in checkpoint_dirs:
            require_duet_eval_artifacts(checkpoint_dir, run_dir, issues)

    utility_root = run_dir / "checkpoint_evals_utility"
    utility_dirs = sorted(
        path
        for path in utility_root.iterdir()
        if utility_root.is_dir() and path.is_dir() and not path.name.startswith("_")
    ) if utility_root.exists() else []
    if not utility_dirs:
        issues.append("checkpoint_evals_utility/*")
    else:
        utility_labels = {path.name for path in utility_dirs}
        if "base_model_orig" not in utility_labels:
            issues.append("checkpoint_evals_utility/base_model_orig/LMEval_SUMMARY.json")
        if "final" not in utility_labels:
            issues.append("checkpoint_evals_utility/final/LMEval_SUMMARY.json")

        checkpoint_labels = {path.name for path in checkpoint_dirs}
        missing_utility_labels = sorted(checkpoint_labels - utility_labels)
        for label in missing_utility_labels:
            issues.append(f"checkpoint_evals_utility/{label}/LMEval_SUMMARY.json")

        for utility_dir in utility_dirs:
            require_file(
                utility_dir / "LMEval_SUMMARY.json",
                f"{utility_dir.relative_to(run_dir)}/LMEval_SUMMARY.json",
                issues,
            )

    return RunCheckResult(run_dir=run_dir, issues=issues)


def main() -> int:
    args = parse_args()
    unlearn_root = resolve_unlearn_root(args.path_to_saves)
    run_dirs = list_run_dirs(unlearn_root)
    expected_runs = build_expected_runs(args.lrs, args.method_variants)

    missing_runs: list[ExpectedRun] = []
    duplicate_runs: list[tuple[ExpectedRun, list[Path]]] = []
    bad_runs: list[RunCheckResult] = []
    matched_run_dirs: list[Path] = []

    for expected in expected_runs:
        matches = find_matches(run_dirs, expected)
        if not matches:
            missing_runs.append(expected)
            continue
        if len(matches) > 1:
            duplicate_runs.append((expected, matches))
            continue

        matched_run_dirs.append(matches[0])
        result = inspect_run_dir(matches[0])
        if result.issues:
            bad_runs.append(result)

    matched_unique = {path.resolve() for path in matched_run_dirs}
    extra_runs = [
        run_dir
        for run_dir in run_dirs
        if any(phase_prefix in run_dir.name for phase_prefix in PHASE_PREFIXES.values())
        and run_dir.resolve() not in matched_unique
    ]

    expected_total = len(expected_runs)
    ok_runs = expected_total - len(missing_runs) - len(duplicate_runs) - len(bad_runs)

    print(f"save_root={unlearn_root}")
    print(f"expected_runs={expected_total}")
    print(f"ok_runs={ok_runs}")
    print(f"missing_runs={len(missing_runs)}")
    print(f"duplicate_matches={len(duplicate_runs)}")
    print(f"broken_runs={len(bad_runs)}")
    print(f"extra_runs={len(extra_runs)}")

    if missing_runs:
        print("\nMissing runs:")
        for item in missing_runs:
            print(f"- phase={item.phase} lr={item.lr} variant={item.variant}")

    if duplicate_runs:
        print("\nDuplicate matches:")
        for item, matches in duplicate_runs:
            print(f"- phase={item.phase} lr={item.lr} variant={item.variant}")
            for match in matches:
                print(f"  {match}")

    if bad_runs:
        print("\nBroken runs:")
        for result in bad_runs:
            print(f"- {result.run_dir}")
            for issue in result.issues:
                print(f"  missing: {issue}")

    if extra_runs:
        print("\nExtra matched-shape runs not covered by current expectation set:")
        for run_dir in extra_runs:
            print(f"- {run_dir}")

    return 0 if not (missing_runs or duplicate_runs or bad_runs) else 1


if __name__ == "__main__":
    sys.exit(main())
