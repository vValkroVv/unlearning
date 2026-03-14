#!/usr/bin/env python3
"""Build a fixed local Utility-1K panel from multiple-choice benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download


DEFAULT_OUTPUT_DIR = Path("artifacts/evals/utility_1k_v1")
DEFAULT_MMLU_PATH = "TIGER-Lab/MMLU-Pro"
DEFAULT_TRUTHFULQA_PATH = "EleutherAI/truthful_qa_binary"
DEFAULT_ARC_PATH = "allenai/ai2_arc"
DEFAULT_WINOGRANDE_PATH = "allenai/winogrande"
DEFAULT_TRUTHFULQA_CONFIG = "multiple_choice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the frozen panel JSONL files and manifests are written.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mmlu-pro", type=int, default=400)
    parser.add_argument("--truthfulqa-bin", type=int, default=200)
    parser.add_argument("--arc", type=int, default=200)
    parser.add_argument("--winogrande", type=int, default=200)
    parser.add_argument(
        "--exclude-targets-file",
        default=None,
        help="Optional path to aliases/targets that should be excluded from the panel.",
    )

    add_source_args(
        parser,
        prefix="mmlu_pro",
        default_path=DEFAULT_MMLU_PATH,
        default_name=None,
        default_split="test",
    )
    add_source_args(
        parser,
        prefix="truthfulqa_bin",
        default_path=DEFAULT_TRUTHFULQA_PATH,
        default_name=DEFAULT_TRUTHFULQA_CONFIG,
        default_split="validation",
    )
    add_source_args(
        parser,
        prefix="arc",
        default_path=DEFAULT_ARC_PATH,
        default_name="ARC-Challenge",
        default_split="validation",
    )
    add_source_args(
        parser,
        prefix="winogrande",
        default_path=DEFAULT_WINOGRANDE_PATH,
        default_name="winogrande_debiased",
        default_split="validation",
    )
    return parser.parse_args()


def add_source_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    default_path: str,
    default_name: str | None,
    default_split: str,
) -> None:
    cli_prefix = prefix.replace("_", "-")
    parser.add_argument(f"--{cli_prefix}-path", default=default_path)
    parser.add_argument(f"--{cli_prefix}-name", default=default_name)
    parser.add_argument(f"--{cli_prefix}-split", default=default_split)
    parser.add_argument(
        f"--{cli_prefix}-data-files",
        default=None,
        help=f"Optional data_files override when loading the {prefix} source.",
    )


def load_source_dataset(
    path: str,
    name: str | None,
    split: str,
    data_files: str | None = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"path": path, "split": split}
    if name not in {None, "", "null", "None"}:
        kwargs["name"] = name
    if data_files not in {None, "", "null", "None"}:
        kwargs["data_files"] = data_files

    try:
        return list(load_dataset(**kwargs))
    except Exception:
        # TruthfulQA-Binary currently needs an explicit dataset script fallback in
        # this workspace's datasets stack.
        if path != DEFAULT_TRUTHFULQA_PATH or data_files not in {None, "", "null", "None"}:
            raise

    script_path = hf_hub_download(
        repo_id=DEFAULT_TRUTHFULQA_PATH,
        filename="truthful_qa_binary.py",
        repo_type="dataset",
    )
    fallback_kwargs = {
        "path": script_path,
        "split": split,
        "trust_remote_code": True,
    }
    if name not in {None, "", "null", "None"}:
        fallback_kwargs["name"] = name
    return list(load_dataset(**fallback_kwargs))


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(collect_strings(item))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(collect_strings(item))
        return strings
    return []


def load_exclusion_aliases(path: str | None) -> set[str]:
    if not path:
        return set()

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    aliases: set[str] = set()
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        for item in collect_strings(payload):
            normalized = normalize_for_match(item)
            if normalized:
                aliases.add(normalized)
        return aliases

    if suffix == ".jsonl":
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    items = collect_strings(payload)
                except json.JSONDecodeError:
                    items = [line]
                for item in items:
                    normalized = normalize_for_match(item)
                    if normalized:
                        aliases.add(normalized)
        return aliases

    with source_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for chunk in re.split(r"[\t,;|]+", line):
                normalized = normalize_for_match(chunk)
                if normalized:
                    aliases.add(normalized)
    return aliases


def has_alias_overlap(text: str, aliases: set[str]) -> bool:
    if not aliases:
        return False
    normalized_text = f" {normalize_for_match(text)} "
    for alias in aliases:
        if f" {alias} " in normalized_text:
            return True
    return False


def filter_overlaps(rows: Iterable[dict[str, Any]], aliases: set[str]) -> tuple[list[dict[str, Any]], int]:
    if not aliases:
        return list(rows), 0

    filtered: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        candidate_text = "\n".join(
            [
                row.get("question", ""),
                *row.get("choices", []),
                row.get("category", ""),
            ]
        )
        if has_alias_overlap(candidate_text, aliases):
            excluded += 1
            continue
        filtered.append(row)
    return filtered, excluded


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} remain after filtering.")
    indices = list(range(len(rows)))
    chosen_indices = sorted(rng.sample(indices, count))
    return [rows[index] for index in chosen_indices]


def proportional_allocation(counts: dict[str, int], target_total: int) -> dict[str, int]:
    available_total = sum(counts.values())
    if available_total < target_total:
        raise ValueError(
            f"Requested {target_total} stratified rows but only {available_total} are available."
        )

    raw_allocations = {
        category: (target_total * category_count) / available_total
        for category, category_count in counts.items()
    }
    allocations = {}
    for category, category_count in counts.items():
        allocations[category] = min(
            category_count,
            math.floor(raw_allocations[category]),
        )
    remainder = target_total - sum(allocations.values())
    if remainder <= 0:
        return allocations

    ranked_categories = sorted(
        counts,
        key=lambda category: (
            raw_allocations[category] - allocations[category],
            counts[category],
            category,
        ),
        reverse=True,
    )
    for category in ranked_categories:
        if remainder == 0:
            break
        if allocations[category] >= counts[category]:
            continue
        allocations[category] += 1
        remainder -= 1
    return allocations


def stratified_sample_by_category(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    allocations = proportional_allocation(
        {category: len(items) for category, items in by_category.items()},
        count,
    )

    selected: list[dict[str, Any]] = []
    realized_allocations: dict[str, int] = {}
    for category in sorted(by_category):
        category_rows = by_category[category]
        allocation = allocations.get(category, 0)
        if allocation == 0:
            continue
        selected.extend(sample_rows(category_rows, allocation, rng))
        realized_allocations[category] = allocation

    return selected, realized_allocations


def convert_mmlu_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_rows):
        rows.append(
            {
                "id": f"mmlu_pro:{row.get('question_id', index)}",
                "source": "mmlu_pro",
                "category": row["category"],
                "question": row["question"],
                "choices": list(row["options"]),
                "gold_idx": int(row["answer_index"]),
            }
        )
    return rows


def convert_truthfulqa_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_rows):
        rows.append(
            {
                "id": f"truthfulqa_binary:{index}",
                "source": "truthfulqa_binary",
                "category": "truthfulness",
                "question": row["question"],
                "choices": list(row["choices"]),
                "gold_idx": int(row["label"]),
            }
        )
    return rows


def convert_arc_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_rows):
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        gold_idx = labels.index(row["answerKey"])
        rows.append(
            {
                "id": f"arc_challenge:{row.get('id', index)}",
                "source": "arc_challenge",
                "category": "science",
                "question": row["question"],
                "choices": choices,
                "gold_idx": gold_idx,
            }
        )
    return rows


def convert_winogrande_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset_rows):
        rows.append(
            {
                "id": f"winogrande:{row.get('id', index)}",
                "source": "winogrande",
                "category": "commonsense",
                "question": row["sentence"],
                "choices": [row["option1"], row["option2"]],
                "gold_idx": int(row["answer"]) - 1,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aliases = load_exclusion_aliases(args.exclude_targets_file)

    rng = random.Random(args.seed)
    stats: dict[str, Any] = {
        "seed": args.seed,
        "output_dir": str(output_dir.resolve()),
        "exclude_targets_file": args.exclude_targets_file,
        "excluded_alias_count": len(aliases),
        "sources": {},
    }

    mmlu_rows = convert_mmlu_rows(
        load_source_dataset(
            path=args.mmlu_pro_path,
            name=args.mmlu_pro_name,
            split=args.mmlu_pro_split,
            data_files=args.mmlu_pro_data_files,
        )
    )
    mmlu_rows, mmlu_excluded = filter_overlaps(mmlu_rows, aliases)
    sampled_mmlu_rows, mmlu_allocations = stratified_sample_by_category(
        mmlu_rows,
        args.mmlu_pro,
        rng,
    )
    mmlu_path = output_dir / f"utility_mmlu_pro_{args.mmlu_pro}.jsonl"
    write_jsonl(mmlu_path, sampled_mmlu_rows)
    stats["sources"]["mmlu_pro"] = {
        "requested": args.mmlu_pro,
        "selected": len(sampled_mmlu_rows),
        "available_after_filter": len(mmlu_rows),
        "excluded_due_overlap": mmlu_excluded,
        "category_counts": mmlu_allocations,
    }

    truthfulqa_rows = convert_truthfulqa_rows(
        load_source_dataset(
            path=args.truthfulqa_bin_path,
            name=args.truthfulqa_bin_name,
            split=args.truthfulqa_bin_split,
            data_files=args.truthfulqa_bin_data_files,
        )
    )
    truthfulqa_rows, truthfulqa_excluded = filter_overlaps(truthfulqa_rows, aliases)
    sampled_truthfulqa_rows = sample_rows(truthfulqa_rows, args.truthfulqa_bin, rng)
    truthfulqa_path = output_dir / f"utility_truthfulqa_bin_{args.truthfulqa_bin}.jsonl"
    write_jsonl(truthfulqa_path, sampled_truthfulqa_rows)
    stats["sources"]["truthfulqa_binary"] = {
        "requested": args.truthfulqa_bin,
        "selected": len(sampled_truthfulqa_rows),
        "available_after_filter": len(truthfulqa_rows),
        "excluded_due_overlap": truthfulqa_excluded,
    }

    arc_rows = convert_arc_rows(
        load_source_dataset(
            path=args.arc_path,
            name=args.arc_name,
            split=args.arc_split,
            data_files=args.arc_data_files,
        )
    )
    arc_rows, arc_excluded = filter_overlaps(arc_rows, aliases)
    sampled_arc_rows = sample_rows(arc_rows, args.arc, rng)
    arc_path = output_dir / f"utility_arc_{args.arc}.jsonl"
    write_jsonl(arc_path, sampled_arc_rows)
    stats["sources"]["arc_challenge"] = {
        "requested": args.arc,
        "selected": len(sampled_arc_rows),
        "available_after_filter": len(arc_rows),
        "excluded_due_overlap": arc_excluded,
    }

    winogrande_rows = convert_winogrande_rows(
        load_source_dataset(
            path=args.winogrande_path,
            name=args.winogrande_name,
            split=args.winogrande_split,
            data_files=args.winogrande_data_files,
        )
    )
    winogrande_rows, winogrande_excluded = filter_overlaps(winogrande_rows, aliases)
    sampled_winogrande_rows = sample_rows(winogrande_rows, args.winogrande, rng)
    winogrande_path = output_dir / f"utility_winogrande_{args.winogrande}.jsonl"
    write_jsonl(winogrande_path, sampled_winogrande_rows)
    stats["sources"]["winogrande"] = {
        "requested": args.winogrande,
        "selected": len(sampled_winogrande_rows),
        "available_after_filter": len(winogrande_rows),
        "excluded_due_overlap": winogrande_excluded,
    }

    manifest = {
        "seed": args.seed,
        "panel_name": "utility_1k",
        "panel_version": output_dir.name,
        "counts": {
            "mmlu_pro": len(sampled_mmlu_rows),
            "truthfulqa_binary": len(sampled_truthfulqa_rows),
            "arc_challenge": len(sampled_arc_rows),
            "winogrande": len(sampled_winogrande_rows),
        },
        "source_configs": {
            "mmlu_pro": {
                "path": args.mmlu_pro_path,
                "name": args.mmlu_pro_name,
                "split": args.mmlu_pro_split,
                "data_files": args.mmlu_pro_data_files,
            },
            "truthfulqa_binary": {
                "path": args.truthfulqa_bin_path,
                "name": args.truthfulqa_bin_name,
                "split": args.truthfulqa_bin_split,
                "data_files": args.truthfulqa_bin_data_files,
            },
            "arc_challenge": {
                "path": args.arc_path,
                "name": args.arc_name,
                "split": args.arc_split,
                "data_files": args.arc_data_files,
            },
            "winogrande": {
                "path": args.winogrande_path,
                "name": args.winogrande_name,
                "split": args.winogrande_split,
                "data_files": args.winogrande_data_files,
            },
        },
        "files": {
            "utility_mmlu_pro": str(mmlu_path.resolve()),
            "utility_truthfulqa_binary": str(truthfulqa_path.resolve()),
            "utility_arc_challenge": str(arc_path.resolve()),
            "utility_winogrande": str(winogrande_path.resolve()),
        },
        "exclude_targets_file": args.exclude_targets_file,
        "excluded_alias_count": len(aliases),
    }

    (output_dir / "panel_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "panel_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
