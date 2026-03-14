#!/usr/bin/env python3
"""Summarize Utility-1K checkpoint evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from checkpoint_summary_utils import collect_eval_summaries


TASK_KEY_MAP = {
    "utility_mmlu_pro_400": "mmlu_pro_400_acc",
    "utility_truthfulqa_bin_200": "truthfulqa_bin_200_acc",
    "utility_arc_200": "arc_200_acc",
    "utility_winogrande_200": "winogrande_200_acc",
}
TASK_COUNT_RE = re.compile(r"_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-name", default="LMEval_SUMMARY.json")
    return parser.parse_args()


def task_weight(task_name: str) -> int:
    match = TASK_COUNT_RE.search(task_name)
    if not match:
        raise ValueError(f"Could not infer sample count from task name: {task_name}")
    return int(match.group(1))


def parse_metric(summary: dict[str, object], task_name: str) -> float | None:
    value = summary.get(f"{task_name}/acc")
    if value is None:
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    base_utility: float | None = None

    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals_utility",
        summary_name=args.summary_name,
    ):
        with Path(row["summary_path"]).open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        output_row = {
            "label": row["label"],
            "step": row["step"],
            "epoch": row["epoch"],
        }
        weighted_sum = 0.0
        total_weight = 0

        for task_name, column_name in TASK_KEY_MAP.items():
            metric_value = parse_metric(metrics, task_name)
            output_row[column_name] = metric_value
            if metric_value is None:
                continue
            weight = task_weight(task_name)
            weighted_sum += metric_value * weight
            total_weight += weight

        utility_avg = weighted_sum / total_weight if total_weight else None
        output_row["utility_avg"] = utility_avg

        if row["label"] == "base_model_orig":
            base_utility = utility_avg
        output_row["utility_delta_vs_base"] = (
            utility_avg - base_utility
            if utility_avg is not None and base_utility is not None
            else None
        )

        rows.append(output_row)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "step",
        "epoch",
        "mmlu_pro_400_acc",
        "truthfulqa_bin_200_acc",
        "arc_200_acc",
        "winogrande_200_acc",
        "utility_avg",
        "utility_delta_vs_base",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
