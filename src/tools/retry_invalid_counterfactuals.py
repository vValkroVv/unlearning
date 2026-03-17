#!/usr/bin/env python3
"""Retry only invalid counterfactual rows and keep the first valid replacement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    dedupe_scored_candidates,
    pick_best_counterfactual_v3,
    read_jsonl,
    save_jsonl,
)
from tools.vllm_cf_client import VLLMCFGenerator, chunked  # noqa: E402


def log(message: str) -> None:
    print(f"[retry_invalid_counterfactuals] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--retry-passes", type=int, default=1)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--generator-concurrency", type=int, default=8)
    parser.add_argument("--generator-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-alternates", type=int, default=1)
    parser.add_argument("--prompt-family", default="default")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def invalid_reason(args, row: Dict[str, Any], alternate: str) -> str | None:
    return counterfactual_invalid_reason(
        clean_counterfactual_text(alternate),
        row.get(args.answer_key, ""),
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )


def build_generator(args) -> VLLMCFGenerator:
    return VLLMCFGenerator(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model=args.vllm_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        concurrency=args.generator_concurrency,
        prompt_family=args.prompt_family,
        num_alternates=args.num_alternates,
    )


def choose_retry_alternate(args, row: Dict[str, Any], response: Dict[str, Any]):
    response_candidates = list(response.get("alternates") or [])
    if not response_candidates and str(response.get("alternate", "")).strip():
        response_candidates = [str(response.get("alternate", "")).strip()]
    candidates, scores = dedupe_scored_candidates(
        response_candidates,
        response.get("scores", []),
    )
    best_alt, pick_meta = pick_best_counterfactual_v3(
        question=str(row.get(args.question_key, "")),
        answer=str(row.get(args.answer_key, "")),
        candidates=candidates,
        candidate_answers=row.get("candidate_answers"),
        external_scores=scores,
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )
    return best_alt, candidates, scores, pick_meta


def main():
    args = parse_args()
    rows = read_jsonl(args.input_path)
    generator = build_generator(args)

    pending: list[int] = []
    for idx, row in enumerate(rows):
        reason = invalid_reason(args, row, str(row.get("alternate", "")))
        row["cf_invalid_reason"] = reason
        row["cf_is_valid"] = reason is None
        row["cf_retry_passes"] = int(row.get("cf_retry_passes", 0))
        if reason is not None:
            pending.append(idx)

    log(f"Loaded rows={len(rows)} invalid_rows={len(pending)} from {args.input_path}")
    recovered = 0

    for retry_pass in range(1, max(0, int(args.retry_passes)) + 1):
        if not pending:
            break

        log(f"Retry pass {retry_pass}: pending_invalid={len(pending)}")
        request_rows: List[Dict[str, Any]] = []
        request_meta: List[int] = []
        for row_idx in pending:
            row = rows[row_idx]
            request_rows.append(
                {
                    "question": str(row[args.question_key]),
                    "answer": str(row[args.answer_key]),
                    "candidate_answers": row.get("candidate_answers"),
                }
            )
            request_meta.append(row_idx)

        next_pending: list[int] = []
        for row_chunk, meta_chunk in zip(
            chunked(request_rows, args.generator_batch_size),
            chunked(request_meta, args.generator_batch_size),
        ):
            outputs = generator.many_sync(list(row_chunk))
            for response, row_idx in zip(outputs, meta_chunk):
                row = dict(rows[row_idx])
                alternate, response_candidates, response_scores, pick_meta = choose_retry_alternate(
                    args,
                    row,
                    response,
                )
                reason = invalid_reason(args, row, alternate)
                row["cf_retry_passes"] = int(row.get("cf_retry_passes", 0)) + 1
                row["cf_retry_last_alternate"] = alternate
                row["cf_retry_last_alternates"] = list(response_candidates)
                row["cf_retry_last_scores"] = list(response_scores)
                row["cf_retry_last_answer_type"] = str(
                    response.get("answer_type", "unknown")
                )
                row["cf_retry_pick_meta"] = pick_meta
                if reason is None:
                    row["alternate"] = alternate
                    row["cf_invalid_reason"] = None
                    row["cf_is_valid"] = True
                    row["cf_source"] = "vllm_retry"
                    row["cf_generator_backend"] = "vllm_openai"
                    row["cf_generator_model"] = args.vllm_model
                    row["cf_same_relation"] = bool(response.get("same_relation", True))
                    row["cf_answer_type"] = str(response.get("answer_type", "unknown"))
                    row["cf_retry_success"] = True
                    recovered += 1
                else:
                    row["cf_invalid_reason"] = reason
                    row["cf_is_valid"] = False
                    row["cf_retry_success"] = False
                    next_pending.append(row_idx)
                rows[row_idx] = row

        pending = next_pending

    log(f"Retry done recovered={recovered} remaining_invalid={len(pending)}")
    save_jsonl(rows, args.output_path)
    log(f"Saved merged raw rows={len(rows)} path={args.output_path}")


if __name__ == "__main__":
    main()
