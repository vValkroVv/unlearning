#!/usr/bin/env python3
"""Merge forget/locality checkpoint metrics with Utility-1K summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from checkpoint_summary_utils import checkpoint_sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-summary", required=True)
    parser.add_argument("--utility-summary", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument(
        "--forget-tau",
        type=float,
        default=None,
        help="Optional forget metric target used to compute U@F_tau.",
    )
    return parser.parse_args()


def load_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def parse_float(value: str | None) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def coalesce(*values):
    for value in values:
        if value not in {None, "", "None"}:
            return value
    return None


def compute_auc(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]["utility_avg"]

    x_key = "epoch" if all(row["epoch"] is not None for row in rows) else "step"
    xs = [row[x_key] for row in rows]
    ys = [row["utility_avg"] for row in rows]
    start = xs[0]
    end = xs[-1]
    if end == start:
        return ys[-1]

    area = 0.0
    for left_x, right_x, left_y, right_y in zip(xs, xs[1:], ys, ys[1:]):
        area += (right_x - left_x) * (left_y + right_y) / 2.0
    return area / (end - start)


def main() -> None:
    args = parse_args()
    checkpoint_rows = load_tsv(Path(args.checkpoint_summary))
    utility_rows = load_tsv(Path(args.utility_summary))

    merged_by_label: dict[str, dict[str, Any]] = {}
    for row in checkpoint_rows:
        label = row["label"]
        merged_by_label[label] = {
            "label": label,
            "checkpoint": row.get("checkpoint"),
            "step": parse_int(row.get("step")),
            "epoch": parse_float(row.get("epoch")),
            "forget_qa_rouge": parse_float(row.get("forget_qa_rouge")),
            "holdout_qa_rouge": parse_float(row.get("holdout_qa_rouge")),
        }

    for row in utility_rows:
        label = row["label"]
        merged = merged_by_label.setdefault(
            label,
            {
                "label": label,
                "checkpoint": label,
                "step": None,
                "epoch": None,
                "forget_qa_rouge": None,
                "holdout_qa_rouge": None,
            },
        )
        merged["checkpoint"] = coalesce(merged.get("checkpoint"), row.get("checkpoint"), label)
        merged["step"] = coalesce(merged.get("step"), parse_int(row.get("step")))
        merged["epoch"] = coalesce(merged.get("epoch"), parse_float(row.get("epoch")))
        for key in (
            "mmlu_pro_400_acc",
            "truthfulqa_bin_200_acc",
            "arc_200_acc",
            "winogrande_200_acc",
            "utility_avg",
            "utility_delta_vs_base",
        ):
            merged[key] = parse_float(row.get(key))

    merged_rows = sorted(
        merged_by_label.values(),
        key=lambda row: checkpoint_sort_key(row["label"], row.get("step")),
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "checkpoint",
        "step",
        "epoch",
        "forget_qa_rouge",
        "holdout_qa_rouge",
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
        for row in merged_rows:
            writer.writerow(row)

    trajectory_rows = [
        row
        for row in merged_rows
        if row.get("utility_avg") is not None and row["label"] != "base_model_run"
    ]
    utility_auc = compute_auc(trajectory_rows)

    running_peak: float | None = None
    max_drawdown = 0.0
    for row in trajectory_rows:
        utility_value = row["utility_avg"]
        if running_peak is None or utility_value > running_peak:
            running_peak = utility_value
        max_drawdown = max(max_drawdown, running_peak - utility_value)

    endpoint_row = trajectory_rows[-1] if trajectory_rows else None
    best_utility = max((row["utility_avg"] for row in trajectory_rows), default=None)
    best_final_gap = (
        best_utility - endpoint_row["utility_avg"]
        if endpoint_row is not None and best_utility is not None
        else None
    )

    u_at_forget_tau = None
    if args.forget_tau is not None:
        matched_rows = [
            row
            for row in trajectory_rows
            if row.get("forget_qa_rouge") is not None and row["label"] != "base_model_orig"
        ]
        if matched_rows:
            best_match = min(
                matched_rows,
                key=lambda row: abs(row["forget_qa_rouge"] - args.forget_tau),
            )
            u_at_forget_tau = {
                "forget_tau": args.forget_tau,
                "label": best_match["label"],
                "forget_qa_rouge": best_match["forget_qa_rouge"],
                "utility_avg": best_match["utility_avg"],
            }

    trajectory_payload = {
        "utility_metric": "utility_avg",
        "utility_auc": utility_auc,
        "utility_max_drawdown": max_drawdown if trajectory_rows else None,
        "utility_best_final_gap": best_final_gap,
        "endpoint_label": endpoint_row["label"] if endpoint_row else None,
        "u_at_forget_tau": u_at_forget_tau,
    }
    trajectory_path = Path(args.trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(
        json.dumps(trajectory_payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
