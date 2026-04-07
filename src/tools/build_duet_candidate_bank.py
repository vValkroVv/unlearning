#!/usr/bin/env python3
"""Build a DUET relation-consistent candidate bank for counterfactual generation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    load_dataset_split,
    normalize_text,
    resolve_answer,
    save_jsonl,
)


def log(message: str) -> None:
    print(f"[build_duet_candidate_bank] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--answer-index", type=int, default=None)
    parser.add_argument("--bucket-key", default="property_pid")
    parser.add_argument("--exclude-id-key", default="object_qid")
    parser.add_argument("--candidates-per-row", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset_split(
        path=args.dataset_path,
        split=args.split,
        name=args.dataset_name,
        data_files=args.data_files,
        max_examples=args.max_examples,
    )
    rows = [dict(row) for row in dataset]
    log(f"Loaded {len(rows)} rows")

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        bucket_value = str(row.get(args.bucket_key, "__missing__"))
        answer = resolve_answer(
            row=row,
            answer_key=args.answer_key,
            answer_index=args.answer_index,
        )
        grouped[bucket_value].append(
            {
                "answer": answer,
                "answer_norm": normalize_text(answer),
                "exclude_id": str(row.get(args.exclude_id_key, "")),
            }
        )

    rng = random.Random(int(args.seed))
    output_rows = []
    candidate_counts = []
    for row in rows:
        answer = resolve_answer(
            row=row,
            answer_key=args.answer_key,
            answer_index=args.answer_index,
        )
        bucket_value = str(row.get(args.bucket_key, "__missing__"))
        exclude_id = str(row.get(args.exclude_id_key, ""))
        answer_norm = normalize_text(answer)

        pool = []
        for candidate in grouped[bucket_value]:
            if candidate["answer_norm"] == answer_norm:
                continue
            if exclude_id and candidate["exclude_id"] and candidate["exclude_id"] == exclude_id:
                continue
            pool.append(candidate["answer"])

        deduped_pool = list(dict.fromkeys(pool))
        if len(deduped_pool) > int(args.candidates_per_row):
            deduped_pool = rng.sample(deduped_pool, k=int(args.candidates_per_row))

        candidate_counts.append(len(deduped_pool))
        output_rows.append(
            {
                "index": int(row["index"]),
                "question": str(row[args.question_key]),
                "answer": answer,
                "candidate_answers": deduped_pool,
                "candidate_relation_scores": [1.0] * len(deduped_pool),
                "candidate_shared_fact_scores": [1.0] * len(deduped_pool),
                "candidate_sources": ["candidate_bank"] * len(deduped_pool),
                "bucket_key": args.bucket_key,
                "bucket_value": bucket_value,
            }
        )

    save_jsonl(output_rows, args.output_path)
    log(f"Saved candidate bank rows={len(output_rows)} path={args.output_path}")

    if args.sidecar_path:
        stats = {
            "rows": len(output_rows),
            "bucket_key": args.bucket_key,
            "candidates_per_row": int(args.candidates_per_row),
            "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
            "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
            "candidate_count_mean": (
                sum(candidate_counts) / float(len(candidate_counts))
                if candidate_counts
                else 0.0
            ),
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar stats to {args.sidecar_path}")


if __name__ == "__main__":
    main()
