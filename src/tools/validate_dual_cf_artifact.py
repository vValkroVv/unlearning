#!/usr/bin/env python3
"""Validate a DualCF JSONL artifact before training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a DualCF JSONL artifact.")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--question-key", choices=("question", "query"), default="question")
    parser.add_argument("--max-bad-rows", type=int, default=10)
    return parser.parse_args()


def normalize_text(value):
    return " ".join(str(value).strip().lower().split())


def main():
    args = parse_args()
    path = Path(args.artifact_path)
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

            if normalize_text(row["answer"]) == normalize_text(row["alternate"]):
                bad_rows.append((line_no, "alternate matches answer after normalization"))

    print(f"artifact={path}")
    print(f"rows={len(seen_indices)}")
    print(f"duplicate_indices={sorted(duplicate_indices)}")
    print(f"bad_rows_count={len(bad_rows)}")
    print(f"bad_rows_sample={bad_rows[: max(0, int(args.max_bad_rows))]}")
    print(
        "ranges="
        + str({key: tuple(value) for key, value in ranges.items() if value[0] != float("inf")})
    )

    if duplicate_indices or bad_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
