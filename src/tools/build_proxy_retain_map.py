#!/usr/bin/env python3
"""Build a syntax-aware retain proxy map for DualCF attribution scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    delex_template,
    load_dataset_split,
    save_jsonl,
    tokenize_normalized_words,
)


def log(message: str) -> None:
    print(f"[build_proxy_retain_map] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forget-dataset-path", required=True)
    parser.add_argument("--forget-split", required=True)
    parser.add_argument("--forget-dataset-name", default=None)
    parser.add_argument("--forget-data-files", default=None)
    parser.add_argument("--retain-dataset-path", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--retain-dataset-name", default=None)
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--forget-question-key", default="question")
    parser.add_argument("--retain-question-key", default=None)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--fallback-top-k", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def jaccard_score(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    if not left or not right:
        return 0.0
    return float(len(left & right)) / float(len(left | right))


def main():
    args = parse_args()
    retain_question_key = args.retain_question_key or args.forget_question_key
    forget_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.forget_dataset_path,
            split=args.forget_split,
            name=args.forget_dataset_name,
            data_files=args.forget_data_files,
            max_examples=args.max_examples,
        )
    ]
    retain_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.retain_dataset_path,
            split=args.retain_split,
            name=args.retain_dataset_name,
            data_files=args.retain_data_files,
        )
    ]
    log(f"Loaded forget_rows={len(forget_rows)} retain_rows={len(retain_rows)}")

    retain_by_template: Dict[str, List[int]] = {}
    retain_tokens: List[Tuple[int, str, List[str]]] = []
    for row in retain_rows:
        template_key = delex_template(row[retain_question_key])
        retain_by_template.setdefault(template_key, []).append(int(row["index"]))
    for template_key, row_indices in retain_by_template.items():
        retain_tokens.append((row_indices[0], template_key, tokenize_normalized_words(template_key)))

    output_rows = []
    exact_matches = 0
    fallback_matches = 0
    candidate_sizes: List[int] = []
    for row in forget_rows:
        template_key = delex_template(row[args.forget_question_key])
        retain_indices = list(retain_by_template.get(template_key, []))
        proxy_mode = "template_exact"
        if retain_indices:
            exact_matches += 1
            retain_indices = retain_indices[: int(args.top_k)]
        else:
            proxy_mode = "template_fallback"
            forget_tokens = tokenize_normalized_words(template_key)
            scored = []
            for first_index, retain_template, retain_template_tokens in retain_tokens:
                scored.append(
                    (
                        jaccard_score(forget_tokens, retain_template_tokens),
                        retain_template,
                    )
                )
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_templates = [
                template for _, template in scored[: int(args.fallback_top_k)] if template
            ]
            retain_indices = []
            for selected_template in selected_templates:
                retain_indices.extend(retain_by_template.get(selected_template, []))
            retain_indices = retain_indices[: int(args.top_k)]
            fallback_matches += 1

        candidate_sizes.append(len(retain_indices))
        output_rows.append(
            {
                "index": int(row["index"]),
                "template_key": template_key,
                "proxy_mode": proxy_mode,
                "retain_indices": retain_indices,
            }
        )

    save_jsonl(output_rows, args.output_path)
    log(f"Saved proxy map rows={len(output_rows)} path={args.output_path}")

    if args.sidecar_path:
        sidecar = {
            "forget_rows": len(forget_rows),
            "retain_rows": len(retain_rows),
            "exact_matches": exact_matches,
            "fallback_matches": fallback_matches,
            "candidate_size_min": min(candidate_sizes) if candidate_sizes else 0,
            "candidate_size_max": max(candidate_sizes) if candidate_sizes else 0,
            "candidate_size_mean": (
                sum(candidate_sizes) / float(len(candidate_sizes))
                if candidate_sizes
                else 0.0
            ),
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar to {args.sidecar_path}")


if __name__ == "__main__":
    main()
