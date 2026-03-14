#!/usr/bin/env python3
"""Validate a DualCF JSONL artifact before training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    BAD_CF_PREFIXES,
    counterfactual_invalid_reason,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a DualCF JSONL artifact.")
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--question-key", choices=("question", "query"), default="question")
    parser.add_argument("--max-bad-rows", type=int, default=10)
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--max-alt-length-chars", type=int, default=None)
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--check-overlap-ratio", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_path = args.input_path or args.artifact_path
    if not artifact_path:
        raise ValueError("Pass --artifact-path or --input-path")
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    required = {
        "index",
        args.question_key,
        "answer",
        "alternate",
        "difficulty_score",
        "attribution_score",
    }
    seen_indices = set()
    duplicate_indices = set()
    bad_rows = []
    ranges = {
        "difficulty_score": [float("inf"), float("-inf")],
        "attribution_score": [float("inf"), float("-inf")],
    }
    optional_numeric_keys = (
        "difficulty_score_raw",
        "attribution_score_raw",
    )
    invalid_reason_counts = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                bad_rows.append((line_no, "empty line"))
                continue

            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                bad_rows.append((line_no, f"missing keys: {missing}"))
                continue

            index = row["index"]
            if index in seen_indices:
                duplicate_indices.add(index)
            seen_indices.add(index)

            for key in (args.question_key, "answer", "alternate"):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad_rows.append((line_no, f"empty or non-string {key}"))

            for score_key in ("difficulty_score", "attribution_score"):
                value = row.get(score_key)
                if not isinstance(value, (int, float)):
                    bad_rows.append((line_no, f"{score_key} is not numeric: {value!r}"))
                    continue
                if not math.isfinite(value):
                    bad_rows.append((line_no, f"{score_key} is not finite: {value!r}"))
                    continue
                ranges[score_key][0] = min(ranges[score_key][0], float(value))
                ranges[score_key][1] = max(ranges[score_key][1], float(value))
                if args.strict and not (0.0 <= float(value) <= 1.0):
                    bad_rows.append((line_no, f"{score_key} out of [0,1]: {value!r}"))

            for score_key in optional_numeric_keys:
                if score_key not in row:
                    continue
                value = row.get(score_key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    bad_rows.append((line_no, f"{score_key} is not finite numeric: {value!r}"))

            invalid_reason = counterfactual_invalid_reason(
                row["alternate"],
                row["answer"],
                reject_gold_substring=args.reject_gold_substring,
                max_overlap_ratio=args.check_overlap_ratio,
                require_short_answer=args.require_short_answer,
                max_alt_length_chars=args.max_alt_length_chars,
            )
            if invalid_reason is not None:
                bad_rows.append((line_no, f"invalid alternate: {invalid_reason}"))
                invalid_reason_counts[invalid_reason] = (
                    invalid_reason_counts.get(invalid_reason, 0) + 1
                )

            alternate_lower = row["alternate"].strip().lower()
            for prefix in BAD_CF_PREFIXES:
                if alternate_lower.startswith(prefix):
                    bad_rows.append((line_no, f"alternate kept banned prefix `{prefix}`"))
                    break

            if args.strict:
                for key in ("difficulty_components", "attribution_components"):
                    if key in row and not isinstance(row[key], dict):
                        bad_rows.append((line_no, f"{key} is not a dict"))

    print(f"artifact={path}")
    print(f"rows={len(seen_indices)}")
    print(f"duplicate_indices={sorted(duplicate_indices)}")
    print(f"bad_rows_count={len(bad_rows)}")
    print(f"bad_rows_sample={bad_rows[: max(0, int(args.max_bad_rows))]}")
    print(f"invalid_reason_counts={invalid_reason_counts}")
    print(
        "ranges="
        + str({key: tuple(value) for key, value in ranges.items() if value[0] != float("inf")})
    )

    if duplicate_indices or bad_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
