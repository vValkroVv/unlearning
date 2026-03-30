#!/usr/bin/env python3
"""Merge local retain fields from a BoundaryCF artifact into a base DualCF/SpanCF artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import read_jsonl, save_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--boundary-artifact-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--copy-boundary-score", action="store_true")
    return parser.parse_args()


def load_rows(path: str):
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON list at {path_obj}, got {type(payload)}")
        return payload
    return read_jsonl(str(path_obj))


def keyed_rows(rows, key_field: str, label: str):
    keyed = {}
    for row_idx, row in enumerate(rows):
        if key_field not in row:
            raise KeyError(f"{label} row {row_idx} is missing `{key_field}`.")
        key = str(row[key_field])
        if key in keyed:
            raise ValueError(f"{label} contains duplicate `{key_field}`={key}.")
        keyed[key] = row
    return keyed


def main():
    args = parse_args()
    base_rows = load_rows(args.base_artifact_path)
    boundary_rows = load_rows(args.boundary_artifact_path)
    boundary_by_key = keyed_rows(boundary_rows, args.mapping_key, "boundary artifact")

    merged_rows = []
    for row_idx, row in enumerate(base_rows):
        if args.mapping_key not in row:
            raise KeyError(f"base artifact row {row_idx} is missing `{args.mapping_key}`.")
        key = str(row[args.mapping_key])
        boundary_row = boundary_by_key.get(key)
        if boundary_row is None:
            raise KeyError(
                f"Missing boundary artifact row for `{args.mapping_key}`={key}."
            )

        updated = dict(row)
        for field in (
            "local_retain_question",
            "local_retain_answer",
            "local_retain_index",
        ):
            if field not in boundary_row:
                raise KeyError(
                    f"boundary artifact row `{args.mapping_key}`={key} is missing `{field}`."
                )
            updated[field] = boundary_row[field]

        if args.copy_boundary_score:
            updated["boundary_score"] = float(boundary_row.get("boundary_score", 0.0))
        else:
            updated["boundary_score"] = 0.0

        merged_rows.append(updated)

    save_jsonl(merged_rows, args.output_path)
    print(
        f"[merge_local_retain_into_artifact] saved rows={len(merged_rows)} path={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
