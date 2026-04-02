#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "src" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from wrong_generation_utils import (  # noqa: E402
    INPUT_TO_OUTPUT_METRIC,
    add_wrong_generation_threshold_args,
    build_wrong_generation_block,
    extract_question,
    has_generation_rows,
)


DUET_EVAL_FILENAME = "DUET_EVAL.json"
WRONG_GENERATIONS_EVAL_FILENAME = "WRONG_GENERATIONS_EVAL.json"
WRONG_GENERATIONS_SUMMARY_FILENAME = "WRONG_GENERATIONS_SUMMARY.json"


def resolve_search_roots(path_to_saves: Path) -> list[Path]:
    root = path_to_saves.expanduser().resolve()
    candidates: list[Path] = []

    if root.name == "unlearn":
        candidates.append(root)
        eval_root = root.parent / "evals"
        if eval_root.exists():
            candidates.append(eval_root)
    elif root.name == "saves":
        unlearn_root = root / "unlearn"
        eval_root = root / "evals"
        if unlearn_root.exists():
            candidates.append(unlearn_root)
        if eval_root.exists():
            candidates.append(eval_root)
        if not candidates:
            candidates.append(root)
    else:
        if (root / "unlearn").exists():
            candidates.append(root / "unlearn")
        if (root / "evals").exists():
            candidates.append(root / "evals")
        if not candidates:
            candidates.append(root)

    return candidates


def collect_eval_paths(path_to_saves: Path) -> list[Path]:
    paths: set[Path] = set()
    for search_root in resolve_search_roots(path_to_saves):
        if not search_root.exists():
            continue
        for path in search_root.rglob(DUET_EVAL_FILENAME):
            if "pretrained" in path.parts:
                continue
            paths.add(path.resolve())
    return sorted(paths)


def enrich_with_questions(
    output_block: dict[str, object],
    value_by_index: dict[str, object],
) -> None:
    output_rows = output_block.get("value_by_index", {})
    if not isinstance(output_rows, dict):
        return

    for index_key, output_row in output_rows.items():
        if not isinstance(output_row, dict):
            continue
        source_row = value_by_index.get(index_key)
        if not isinstance(source_row, dict):
            source_row = value_by_index.get(str(index_key))
        if not isinstance(source_row, dict):
            continue

        output_row["index"] = str(index_key)
        question = extract_question(str(source_row.get("input", "") or ""))
        if question:
            output_row["question"] = question


def process_file(path: Path, args: argparse.Namespace) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    output_payload: dict[str, object] = {}
    summary_payload: dict[str, float] = {}
    updated = False

    for input_metric, output_metric in INPUT_TO_OUTPUT_METRIC.items():
        metric_payload = data.get(input_metric)
        if not isinstance(metric_payload, dict):
            continue
        value_by_index = metric_payload.get("value_by_index", {})
        if not has_generation_rows(value_by_index):
            continue

        output_block = build_wrong_generation_block(value_by_index, args)
        enrich_with_questions(output_block, value_by_index)
        output_payload[output_metric] = output_block
        summary_payload[output_metric] = float(output_block["agg_value"])
        updated = True

    if not updated:
        return False

    output_path = path.parent / WRONG_GENERATIONS_EVAL_FILENAME
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    summary_path = path.parent / WRONG_GENERATIONS_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_saves",
        type=Path,
        required=True,
        help="Path to saves/ or saves/unlearn",
    )
    add_wrong_generation_threshold_args(parser)
    args = parser.parse_args()

    total = 0

    try:
        from tqdm import tqdm  # type: ignore

        def _iter(items: Iterable[Path], desc: str):
            return tqdm(list(items), desc=desc, unit="file")
    except Exception:
        def _iter(items: Iterable[Path], desc: str):
            print(f"[wrong_gen] {desc}")
            return items

    paths = collect_eval_paths(args.path_to_saves)
    if not paths:
        print(f"[wrong_gen] No {DUET_EVAL_FILENAME} files found under {args.path_to_saves}")
        print(
            "[wrong_gen] Expected to find files like evals/DUET_EVAL.json "
            "and checkpoint_evals/checkpoint-*/DUET_EVAL.json"
        )
        return

    print(f"[wrong_gen] Found {len(paths)} {DUET_EVAL_FILENAME} files under {args.path_to_saves}")
    for path in _iter(paths, "wrong_gen"):
        print(f"[wrong_gen] Processing {path}")
        if process_file(path, args):
            total += 1
        else:
            print(f"[wrong_gen] Skipped (no mapped generation metrics): {path}")

    print(f"Written {WRONG_GENERATIONS_EVAL_FILENAME} for {total} eval folders.")


if __name__ == "__main__":
    main()
