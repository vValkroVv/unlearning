#!/usr/bin/env python3
"""Score an explicit rarity controller for DualCF artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import load_dataset_split, save_jsonl


def log(message: str) -> None:
    print(f"[score_rarity] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--popularity-column", default="pop_sum")
    parser.add_argument("--q-low", type=float, default=0.05)
    parser.add_argument("--q-high", type=float, default=0.95)
    parser.add_argument("--reference-dataset-path", default=None)
    parser.add_argument("--reference-dataset-name", default=None)
    parser.add_argument("--reference-data-files", default=None)
    parser.add_argument("--reference-split", default=None)
    parser.add_argument("--reference-splits", nargs="+", default=None)
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def _normalize_optional_arg(value: Optional[str]) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _load_rows(
    *,
    dataset_path: str,
    split: str,
    dataset_name: Optional[str],
    data_files: Optional[str],
):
    return [
        dict(row)
        for row in load_dataset_split(
            path=dataset_path,
            split=split,
            name=dataset_name,
            data_files=data_files,
        )
    ]


def _coerce_popularity(
    row: dict,
    *,
    popularity_column: str,
    row_position: int,
) -> float:
    if popularity_column not in row:
        raise KeyError(
            f"Missing popularity column `{popularity_column}` at "
            f"row_position={row_position} index={row.get('index', '<missing>')}"
        )
    value = row[popularity_column]
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(
            f"Non-numeric popularity `{popularity_column}`={value!r} at "
            f"row_position={row_position} index={row.get('index', '<missing>')}"
        )
    value = float(value)
    if value < 0.0:
        raise ValueError(
            f"Negative popularity `{popularity_column}`={value!r} at "
            f"row_position={row_position} index={row.get('index', '<missing>')}"
        )
    return value


def _quantile(values: Iterable[float], q: float) -> float:
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    if tensor.numel() <= 0:
        raise ValueError("Cannot compute quantiles on an empty reference population.")
    return float(torch.quantile(tensor, q).item())


def _reference_splits(args) -> list[str]:
    reference_split = _normalize_optional_arg(args.reference_split)
    if reference_split is not None:
        splits = [split.strip() for split in reference_split.split("+") if split.strip()]
        if splits:
            return splits
    if args.reference_splits:
        splits = [
            split
            for split in args.reference_splits
            if _normalize_optional_arg(split) is not None
        ]
        if splits:
            return splits
    if args.split in (None, "", "null", "None"):
        raise ValueError("Pass --reference-splits when no primary --split is available.")
    return [str(args.split)]


def main():
    args = parse_args()
    if not (0.0 <= float(args.q_low) < float(args.q_high) <= 1.0):
        raise ValueError("Require 0 <= q_low < q_high <= 1.")

    if args.input_path not in (None, "", "null", "None"):
        args.dataset_path = "json"
        args.split = "train"
        args.data_files = args.input_path
    if args.dataset_path in (None, "", "null", "None") or args.split in (
        None,
        "",
        "null",
        "None",
    ):
        raise ValueError("Provide --input-path or both --dataset-path and --split")

    dataset_path = str(args.dataset_path)
    split = str(args.split)
    dataset_name = _normalize_optional_arg(args.dataset_name)
    data_files = _normalize_optional_arg(args.data_files)
    popularity_column = str(args.popularity_column)

    reference_dataset_path = (
        _normalize_optional_arg(args.reference_dataset_path) or dataset_path
    )
    reference_dataset_name = (
        _normalize_optional_arg(args.reference_dataset_name)
        if args.reference_dataset_name is not None
        else dataset_name
    )
    reference_data_files_arg = _normalize_optional_arg(args.reference_data_files)
    # Only inherit the primary data_files when the reference source is the same
    # dataset source. When the reference source points to a different HF/local
    # dataset, carrying over the artifact jsonl path makes datasets try to merge
    # incompatible schemas.
    if reference_data_files_arg is not None:
        reference_data_files = reference_data_files_arg
    elif (
        reference_dataset_path == dataset_path
        and reference_dataset_name == dataset_name
    ):
        reference_data_files = data_files
    else:
        reference_data_files = None
    reference_splits = _reference_splits(args)

    log(
        "Starting with "
        f"dataset_path={dataset_path} split={split} "
        f"reference_dataset_path={reference_dataset_path} "
        f"reference_splits={reference_splits} "
        f"reference_data_files={reference_data_files} "
        f"output_path={args.output_path}"
    )

    rows = _load_rows(
        dataset_path=dataset_path,
        split=split,
        dataset_name=dataset_name,
        data_files=data_files,
    )
    if not rows:
        raise ValueError("Input dataset is empty.")

    reference_popularity = []
    for reference_split in reference_splits:
        reference_rows = _load_rows(
            dataset_path=reference_dataset_path,
            split=reference_split,
            dataset_name=reference_dataset_name,
            data_files=reference_data_files,
        )
        for row_position, row in enumerate(reference_rows):
            pop_sum = _coerce_popularity(
                row,
                popularity_column=popularity_column,
                row_position=row_position,
            )
            reference_popularity.append(math.log1p(pop_sum))

    if not reference_popularity:
        raise ValueError("Reference population is empty.")

    q_low_value = _quantile(reference_popularity, float(args.q_low))
    q_high_value = _quantile(reference_popularity, float(args.q_high))
    denom = q_high_value - q_low_value
    degenerate_reference = math.isclose(denom, 0.0, rel_tol=0.0, abs_tol=1e-12)
    if degenerate_reference:
        log(
            "Reference quantiles collapsed; all rows will receive "
            "rarity_score=0.0 to disable rarity routing."
        )

    output_rows = []
    rarity_values = []
    for row_position, row in enumerate(rows):
        pop_sum = _coerce_popularity(
            row,
            popularity_column=popularity_column,
            row_position=row_position,
        )
        z_value = math.log1p(pop_sum)
        if degenerate_reference:
            rarity = 0.0
        else:
            popularity_norm = (z_value - q_low_value) / denom
            popularity_norm = max(0.0, min(1.0, float(popularity_norm)))
            rarity = 1.0 - popularity_norm

        updated = dict(row)
        updated["rarity_score_raw"] = float(rarity)
        updated["rarity_score"] = float(rarity)
        updated["rarity_recipe"] = {
            "mode": "log_quantile",
            "q_low": float(args.q_low),
            "q_high": float(args.q_high),
            "popularity_column": popularity_column,
            "reference_dataset": reference_dataset_path,
            "reference_dataset_name": reference_dataset_name,
            "reference_split": "+".join(reference_splits),
            "reference_data_files": reference_data_files,
        }
        output_rows.append(updated)
        rarity_values.append(float(rarity))

    save_jsonl(output_rows, args.output_path)
    log(
        "Saved "
        f"rows={len(output_rows)} rarity_score_range="
        f"({min(rarity_values):.6f}, {max(rarity_values):.6f}) "
        f"path={args.output_path}"
    )

    if args.sidecar_path:
        sidecar = {
            "rows": len(output_rows),
            "mode": "log_quantile",
            "q_low": float(args.q_low),
            "q_high": float(args.q_high),
            "reference_dataset": reference_dataset_path,
            "reference_dataset_name": reference_dataset_name,
            "reference_split": "+".join(reference_splits),
            "reference_data_files": reference_data_files,
            "reference_count": len(reference_popularity),
            "reference_z_min": min(reference_popularity),
            "reference_z_max": max(reference_popularity),
            "reference_q_low": q_low_value,
            "reference_q_high": q_high_value,
            "degenerate_reference": degenerate_reference,
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar to {args.sidecar_path}")


if __name__ == "__main__":
    main()
