#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import load_dataset_split


def log(message: str) -> None:
    print(f"[build_duet_merged_sidecar_from_parts] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic merged DUET sidecar from existing rare and popular "
            "sidecars by concatenating the already generated source rows."
        )
    )
    parser.add_argument("--dataset-path", default="SwetieePawsss/DUET")
    parser.add_argument("--rare-dir", required=True)
    parser.add_argument("--popular-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rare-split", default="city_forget_rare_5")
    parser.add_argument("--popular-split", default="city_forget_popular_5")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--answer-index", type=int, default=None)
    parser.add_argument("--sidecar-name", default="api_sidecar.jsonl")
    parser.add_argument("--merged-data-name", default="merged_input.jsonl")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fail(message: str) -> None:
    raise RuntimeError(message)


def summarize_sidecar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[int] = set()
    duplicates: list[int] = []
    bad_rows = 0
    for row in rows:
        idx = int(row["index"])
        if idx in seen:
            duplicates.append(idx)
        seen.add(idx)
        alternates = row.get("alternates")
        if not isinstance(alternates, list) or not alternates:
            bad_rows += 1
            continue
        for key in (
            "scores",
            "relation_scores",
            "shared_fact_scores",
            "candidate_sources",
        ):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != len(alternates):
                bad_rows += 1
                break
    return {
        "rows": len(rows),
        "unique_indices": len(seen),
        "duplicate_indices": duplicates,
        "bad_rows": bad_rows,
    }


def validate_source_sidecar(
    sidecar_rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    ordered = sorted(sidecar_rows, key=lambda row: int(row["index"]))
    expected_index = 0
    for row in ordered:
        idx = int(row["index"])
        if idx != expected_index:
            fail(
                f"{source_name} sidecar indices must be contiguous from 0. "
                f"Expected {expected_index}, got {idx}."
            )
        expected_index += 1
        alternates = row.get("alternates")
        if not isinstance(alternates, list) or not alternates:
            fail(f"{source_name} sidecar index={idx} has empty alternates.")
        for key in (
            "scores",
            "relation_scores",
            "shared_fact_scores",
            "candidate_sources",
        ):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != len(alternates):
                fail(
                    f"{source_name} sidecar index={idx} has misaligned `{key}` metadata."
                )
    return ordered


def load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing metadata file: {path}")
    return read_json(path)


def assert_compatible_metas(rare_meta: dict[str, Any], popular_meta: dict[str, Any]) -> None:
    for key in (
        "backend",
        "model",
        "input_models",
        "input_reasoning_efforts",
        "prompt_family",
        "dataset_path",
        "dataset_name",
        "question_key",
        "answer_key",
        "answer_index",
        "num_alternates",
        "batch_size",
        "reasoning_effort",
        "temperature",
        "top_p",
        "max_completion_tokens",
    ):
        if rare_meta.get(key) != popular_meta.get(key):
            fail(
                f"Rare/popular sidecar metadata mismatch for `{key}`: "
                f"{rare_meta.get(key)!r} != {popular_meta.get(key)!r}"
            )


def load_source_rows(
    *,
    dataset_path: str,
    split: str,
    count: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset_split(
        path=dataset_path,
        split=split,
        max_examples=count,
    )
    rows = [dict(row) for row in dataset]
    if len(rows) != count:
        fail(f"Loaded {len(rows)} rows for split={split}, expected {count}.")
    return rows


def main() -> None:
    args = parse_args()
    rare_dir = Path(args.rare_dir)
    popular_dir = Path(args.popular_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rare_sidecar_path = rare_dir / args.sidecar_name
    popular_sidecar_path = popular_dir / args.sidecar_name
    output_sidecar_path = output_dir / args.sidecar_name
    merged_data_path = output_dir / args.merged_data_name

    rare_rows = validate_source_sidecar(
        read_jsonl(rare_sidecar_path),
        source_name="rare",
    )
    popular_rows = validate_source_sidecar(
        read_jsonl(popular_sidecar_path),
        source_name="popular",
    )
    rare_meta = load_meta(rare_dir / f"{args.sidecar_name}.meta.json")
    popular_meta = load_meta(popular_dir / f"{args.sidecar_name}.meta.json")
    assert_compatible_metas(rare_meta, popular_meta)

    rare_source_rows = load_source_rows(
        dataset_path=args.dataset_path,
        split=args.rare_split,
        count=len(rare_rows),
    )
    popular_source_rows = load_source_rows(
        dataset_path=args.dataset_path,
        split=args.popular_split,
        count=len(popular_rows),
    )
    combined_source_rows = rare_source_rows + popular_source_rows
    repeated_signatures = Counter(
        json.dumps(
            {key: row[key] for key in sorted(row) if key != "index"},
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in combined_source_rows
    )
    duplicate_signatures = sum(1 for count in repeated_signatures.values() if count > 1)
    if duplicate_signatures:
        fail(
            f"Rare/popular source rows are not unique enough to build a synthetic merge. "
            f"duplicate_signatures={duplicate_signatures}"
        )

    merged_input_rows: list[dict[str, Any]] = []
    for new_index, (source_row, split_label) in enumerate(
        zip(
            combined_source_rows,
            ["rare"] * len(rare_source_rows) + ["popular"] * len(popular_source_rows),
        )
    ):
        updated = dict(source_row)
        updated["index"] = int(new_index)
        updated["source_split_label"] = split_label
        merged_input_rows.append(updated)

    with merged_data_path.open("w", encoding="utf-8") as handle:
        for row in merged_input_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    merged_rows: list[dict[str, Any]] = []
    source_sidecar_rows = rare_rows + popular_rows
    for merged_row, source_sidecar_row, split_label in zip(
        merged_input_rows,
        source_sidecar_rows,
        ["rare"] * len(rare_rows) + ["popular"] * len(popular_rows),
    ):
        updated = dict(source_sidecar_row)
        updated["index"] = int(merged_row["index"])
        updated["source_split_label"] = split_label
        updated["source_split_index"] = int(source_sidecar_row["index"])
        merged_rows.append(updated)

    with output_sidecar_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_sidecar(merged_rows)
    summary.update(
        {
            "source_rows_rare": len(rare_rows),
            "source_rows_popular": len(popular_rows),
            "merged_rows": len(merged_rows),
            "synthetic_order": "rare_then_popular",
            "duplicate_row_signatures": duplicate_signatures,
            "merged_data_file": str(merged_data_path),
        }
    )
    summary_path = output_dir / f"{args.sidecar_name}.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    merged_meta = dict(rare_meta)
    merged_meta.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": "synthetic_rare_then_popular",
            "data_files": str(merged_data_path),
            "max_examples": len(merged_rows),
            "candidate_bank": str(output_dir / "step0_candidate_bank.jsonl"),
            "source_rare_dir": str(rare_dir),
            "source_popular_dir": str(popular_dir),
            "source_rows_rare": len(rare_rows),
            "source_rows_popular": len(popular_rows),
            "derived_from_parts": True,
            "derived_strategy": "synthetic_dataset_rare_then_popular_reindex",
            "synthetic_order": "rare_then_popular",
            "duplicate_row_signatures": duplicate_signatures,
            "merged_data_file": str(merged_data_path),
        }
    )
    meta_path = output_dir / f"{args.sidecar_name}.meta.json"
    meta_path.write_text(
        json.dumps(merged_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(
        f"wrote rows={len(merged_rows)} output={output_sidecar_path} "
        f"rare_rows={len(rare_rows)} popular_rows={len(popular_rows)} "
        f"merged_data={merged_data_path}"
    )


if __name__ == "__main__":
    main()
