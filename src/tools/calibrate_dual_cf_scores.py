#!/usr/bin/env python3
"""Calibrate DualCF routing scores offline before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import percentile_rank, read_jsonl, save_jsonl


def log(message: str) -> None:
    print(f"[calibrate_dual_cf_scores] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--difficulty-in", default="difficulty_score_raw")
    parser.add_argument("--difficulty-out", default="difficulty_score")
    parser.add_argument("--attribution-in", default="attribution_score_raw")
    parser.add_argument("--attribution-out", default="attribution_score")
    parser.add_argument("--method", choices=("percentile",), default="percentile")
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if not rows:
        raise ValueError("Input artifact is empty")

    if args.method != "percentile":
        raise NotImplementedError(args.method)

    difficulty_values = [float(row[args.difficulty_in]) for row in rows]
    attribution_values = [float(row[args.attribution_in]) for row in rows]
    difficulty_pct = percentile_rank(difficulty_values)
    attribution_pct = percentile_rank(attribution_values)

    for row, difficulty_score, attribution_score in zip(
        rows, difficulty_pct, attribution_pct
    ):
        row[args.difficulty_out] = float(difficulty_score)
        row[args.attribution_out] = float(attribution_score)
        row["routing_calibration"] = args.method

    save_jsonl(rows, args.output_path)
    log(f"Saved calibrated rows={len(rows)} path={args.output_path}")

    if args.sidecar_path:
        sidecar = {
            "rows": len(rows),
            "method": args.method,
            "difficulty_in": args.difficulty_in,
            "difficulty_out": args.difficulty_out,
            "attribution_in": args.attribution_in,
            "attribution_out": args.attribution_out,
            "difficulty_raw_min": min(difficulty_values),
            "difficulty_raw_max": max(difficulty_values),
            "attribution_raw_min": min(attribution_values),
            "attribution_raw_max": max(attribution_values),
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar to {args.sidecar_path}")


if __name__ == "__main__":
    main()
