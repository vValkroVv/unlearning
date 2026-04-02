#!/usr/bin/env python3
"""Shared wrong-generation heuristics for DUET-style eval logs."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any


QUESTION_RE = re.compile(r"\buser\b\s*\n\n(?P<question>.*?)\bassistant\b\s*\n*\Z", re.DOTALL)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PROMPT_MARKERS = ("system\n", "user\n", "assistant\n")
REASON_ORDER = [
    "empty_like",
    "prompt_leak",
    "punctuation_spam",
    "char_repeat",
    "token_repeat",
    "ngram_repeat",
    "low_diversity",
    "too_long",
]
INPUT_TO_OUTPUT_METRIC = {
    "forget_qa_rouge": "forget_wrong_gen_rate",
    "holdout_qa_rouge": "holdout_wrong_gen_rate",
}


def add_wrong_generation_threshold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--punctuation-ratio-threshold",
        type=float,
        default=0.58,
        help=(
            "Flag punctuation spam when punctuation ratio crosses this value and "
            "the generation has at least four word tokens."
        ),
    )
    parser.add_argument(
        "--char-run-threshold",
        type=int,
        default=8,
        help="Flag repeated characters when the longest identical-character run reaches this length.",
    )
    parser.add_argument(
        "--token-run-threshold",
        type=int,
        default=6,
        help="Flag repeated tokens when the longest identical-token run reaches this length.",
    )
    parser.add_argument(
        "--bigram-repeat-threshold",
        type=int,
        default=8,
        help="Flag repeated 2-grams when the most common 2-gram appears at least this many times.",
    )
    parser.add_argument(
        "--trigram-repeat-threshold",
        type=int,
        default=6,
        help="Flag repeated 3-grams when the most common 3-gram appears at least this many times.",
    )
    parser.add_argument(
        "--low-diversity-min-words",
        type=int,
        default=12,
        help="Require at least this many word tokens before low-diversity checks can trigger.",
    )
    parser.add_argument(
        "--dominant-token-ratio-threshold",
        type=float,
        default=0.50,
        help="Flag low diversity when one token accounts for at least this fraction of word tokens.",
    )
    parser.add_argument(
        "--unique-token-ratio-threshold",
        type=float,
        default=0.35,
        help="Flag low diversity when the unique-token ratio drops to or below this value.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=16,
        help="Absolute word-count threshold for overlong generations.",
    )
    parser.add_argument(
        "--relative-word-multiplier",
        type=float,
        default=4.0,
        help="Relative overlong threshold multiplier applied to ground-truth word count.",
    )
    parser.add_argument(
        "--relative-word-margin",
        type=int,
        default=4,
        help="Extra slack added after applying the relative overlong threshold.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=160,
        help="Absolute character-count threshold for overlong generations.",
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def dominant_ngram_count(tokens: list[str], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return counts.most_common(1)[0][1]


def has_generation_rows(value_by_index: Any) -> bool:
    if not isinstance(value_by_index, dict):
        return False
    return any(
        isinstance(row, dict) and "generation" in row
        for row in value_by_index.values()
    )


def extract_question(prompt: str) -> str:
    if not prompt:
        return ""
    normalized = prompt.replace("\r\n", "\n").strip()
    match = QUESTION_RE.search(normalized)
    if match is not None:
        return match.group("question").strip()
    user_marker = normalized.rfind("user\n\n")
    if user_marker != -1:
        question = normalized[user_marker + len("user\n\n") :]
        assistant_marker = question.rfind("assistant")
        if assistant_marker != -1:
            question = question[:assistant_marker]
        question = question.strip()
        if question:
            return question
    return normalized


def classify_generation(
    generation: str,
    ground_truth: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    text = str(generation or "").strip()
    gt_text = str(ground_truth or "").strip()
    tokens = tokenize(text)
    word_tokens = [token for token in tokens if re.search(r"\w", token)]
    gt_word_tokens = [token for token in tokenize(gt_text) if re.search(r"\w", token)]
    non_space_chars = [char for char in text if not char.isspace()]
    punct_count = sum(1 for char in non_space_chars if not char.isalnum())
    punct_ratio = punct_count / max(1, len(non_space_chars))

    longest_char_run = 0
    current_run = 0
    previous_char = None
    for char in text:
        if char == previous_char:
            current_run += 1
        else:
            current_run = 1
            previous_char = char
        longest_char_run = max(longest_char_run, current_run)

    longest_token_run = 0
    current_run = 0
    previous_token = None
    for token in tokens:
        if token == previous_token:
            current_run += 1
        else:
            current_run = 1
            previous_token = token
        longest_token_run = max(longest_token_run, current_run)

    dominant_token_count = Counter(word_tokens).most_common(1)[0][1] if word_tokens else 0
    dominant_token_ratio = dominant_token_count / max(1, len(word_tokens))
    unique_token_ratio = (
        len(set(word_tokens)) / max(1, len(word_tokens)) if word_tokens else 0.0
    )
    bigram_repeat = dominant_ngram_count(tokens, 2)
    trigram_repeat = dominant_ngram_count(tokens, 3)
    too_long_limit = max(
        args.max_words,
        int(len(gt_word_tokens) * args.relative_word_multiplier + args.relative_word_margin),
    )

    reasons: set[str] = set()
    if not text or not any(char.isalnum() for char in text):
        reasons.add("empty_like")
    if any(marker in text.lower() for marker in PROMPT_MARKERS):
        reasons.add("prompt_leak")
    if len(word_tokens) >= 4 and punct_ratio >= args.punctuation_ratio_threshold:
        reasons.add("punctuation_spam")
    if longest_char_run >= args.char_run_threshold:
        reasons.add("char_repeat")
    if longest_token_run >= args.token_run_threshold:
        reasons.add("token_repeat")
    if (
        bigram_repeat >= args.bigram_repeat_threshold
        or trigram_repeat >= args.trigram_repeat_threshold
    ):
        reasons.add("ngram_repeat")
    if (
        len(word_tokens) >= args.low_diversity_min_words
        and dominant_token_ratio >= args.dominant_token_ratio_threshold
    ):
        reasons.add("low_diversity")
    if (
        len(word_tokens) >= args.low_diversity_min_words
        and unique_token_ratio <= args.unique_token_ratio_threshold
    ):
        reasons.add("low_diversity")
    if len(word_tokens) >= too_long_limit or len(text) >= args.max_chars:
        reasons.add("too_long")

    features = {
        "word_count": len(word_tokens),
        "char_count": len(text),
        "ground_truth_word_count": len(gt_word_tokens),
        "punctuation_ratio": punct_ratio,
        "longest_char_run": longest_char_run,
        "longest_token_run": longest_token_run,
        "bigram_repeat": bigram_repeat,
        "trigram_repeat": trigram_repeat,
        "dominant_token_ratio": dominant_token_ratio,
        "unique_token_ratio": unique_token_ratio,
        "too_long_limit": too_long_limit,
    }
    ordered_reasons = [reason for reason in REASON_ORDER if reason in reasons]
    return ordered_reasons, features


def build_wrong_generation_block(
    value_by_index: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_by_index: dict[str, Any] = {}
    wrong_count = 0
    total_count = 0
    reason_counts: Counter[str] = Counter()

    for index_key, row in value_by_index.items():
        if not isinstance(row, dict) or "generation" not in row:
            continue
        generation = str(row.get("generation", "") or "")
        ground_truth = str(row.get("ground_truth", "") or "")
        reasons, features = classify_generation(generation, ground_truth, args)
        is_wrong = bool(reasons)

        total_count += 1
        if is_wrong:
            wrong_count += 1
            reason_counts.update(reasons)

        payload = {
            "wrong_generation": is_wrong,
            "reasons": reasons,
            "ground_truth": ground_truth,
            "generation": generation,
            **features,
        }
        if "rougeL_recall" in row:
            payload["rougeL_recall"] = row.get("rougeL_recall")
        output_by_index[str(index_key)] = payload

    agg_value = float(wrong_count / total_count) if total_count else 0.0
    return {
        "agg_value": agg_value,
        "wrong_count": wrong_count,
        "total_count": total_count,
        "reason_counts": {
            reason: reason_counts.get(reason, 0)
            for reason in REASON_ORDER
            if reason_counts.get(reason, 0) > 0
        },
        "value_by_index": output_by_index,
    }
