#!/usr/bin/env python3
"""Analyze malformed or overlong generations in DUET_EVAL-style logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from checkpoint_summary_utils import collect_eval_summaries, parse_checkpoint_step
from new_method_variant_utils import extract_new_method_variant, variant_info_from_method_key, variant_sort_key


DUET_EVAL_FILENAME = "DUET_EVAL.json"
DUET_SUMMARY_FILENAME = "DUET_SUMMARY.json"
LR_RE = re.compile(r"_lr([^_]+)")
SEED_RE = re.compile(r"_seed(\d+)")
METHOD_RE = re.compile(
    r"_(dual_cf|dpo_cf|ga|ada_pop|loku|npo_sam|npo|simnpo|simple_ce|multicf|boundary_cf|span_cf_simnpo_local_retain|span_cf_simnpo_projected|span_cf_simnpo_sam|span_cf_local_retain|span_cf_simnpo|span_cf|falcon)_lora_.*?_lr[^_]+(.*)$"
)
DUAL_FLAG_RE = re.compile(r"^(dOn|dOff|aOn|aOff|adT|adF)$")
RUN_SPLIT_PATTERNS = [
    r"_city_forget_rare_5_",
    r"_city_forget_popular_5_",
    r"_city_forget_5_",
    r"_forget_level\d+_",
]
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
METHOD_DISPLAY = {
    "ga": "GA",
    "ada_pop": "AdaPop",
    "npo": "NPO",
    "npo_sam": "NPO-SAM",
    "loku": "LoKU",
    "dual_cf": "DualCF",
    "dual_cf_full": "DualCF(full)",
    "dual_cf_d_only": "DualCF(d_only)",
    "dual_cf_a_only": "DualCF(a_only)",
    "dpo_cf": "DPO-CF",
    "simnpo": "SimNPO",
    "simple_ce": "Simple-CE",
    "span_cf_simnpo": "SpanCF-SimNPO",
    "span_cf_local_retain": "SpanCF-LocalRetain",
    "span_cf_simnpo_local_retain": "SpanCF-SimNPO-LocalRetain",
    "span_cf_simnpo_sam": "SpanCF-SimNPO-SAM",
    "span_cf_simnpo_projected": "SpanCF-SimNPO-Projected",
    "falcon": "FALCON",
}
METHOD_ORDER = [
    "ga",
    "ada_pop",
    "npo",
    "npo_sam",
    "loku",
    "dual_cf_full",
    "dual_cf_d_only",
    "dual_cf_a_only",
    "dual_cf",
    "dpo_cf",
    "simnpo",
    "simple_ce",
    "span_cf_simnpo",
    "span_cf_local_retain",
    "span_cf_simnpo_local_retain",
    "span_cf_simnpo_sam",
    "span_cf_simnpo_projected",
    "falcon",
]
METHOD_ORDER_INDEX = {name: index for index, name in enumerate(METHOD_ORDER)}


@dataclass(frozen=True)
class EvalLog:
    input_root: Path
    input_root_label: str
    eval_path: Path
    run_dir: Path
    run_name: str
    benchmark: str
    model_label: str
    forget_split: str
    holdout_split: str
    lr: str
    seed: str
    method_key: str
    method_display: str
    stage_label: str
    step: int | None
    epoch: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Root to scan. Can point at part1/part2 folders, extracted saves, or a specific run tree.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where TSV summaries and sampled examples will be written.",
    )
    parser.add_argument(
        "--eval-filename",
        default=DUET_EVAL_FILENAME,
        help="Eval filename to scan for. Defaults to DUET_EVAL.json.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="How many wrong and clean samples to keep per root/method/metric bucket.",
    )
    parser.add_argument(
        "--punctuation-ratio-threshold",
        type=float,
        default=0.58,
        help="Flag punctuation spam when punctuation ratio crosses this value and the generation has at least four word tokens.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing results.",
    )
    return parser.parse_args()


def prepare_output_root(path: Path, overwrite: bool) -> Path:
    root = path.expanduser().resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {root}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_path(path: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(cwd))
    except ValueError:
        return str(path.resolve())


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{100.0 * numerator / denominator:.2f}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    slug = slug.strip("-")
    return slug or "unknown"


def infer_input_root_label(root: Path) -> str:
    if root.name:
        return root.name
    return slugify(str(root))


def extract_lr(run_name: str) -> str:
    match = LR_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse lr from run name: {run_name}")
    return match.group(1)


def extract_seed(run_name: str) -> str:
    match = SEED_RE.search(run_name)
    if match is None:
        return ""
    return match.group(1)


def extract_method_key(run_name: str) -> str:
    match = METHOD_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse method from run name: {run_name}")
    method_name = match.group(1)
    suffix = match.group(2)
    flags = [token for token in suffix.split("_") if DUAL_FLAG_RE.fullmatch(token)]

    if method_name == "dual_cf":
        flag_set = set(flags)
        if flag_set == {"dOn", "aOn"}:
            return "dual_cf_full"
        if flag_set == {"dOn", "aOff"}:
            return "dual_cf_d_only"
        if flag_set == {"dOff", "aOn"}:
            return "dual_cf_a_only"
        if flags:
            return "dual_cf_" + "_".join(flags)
        return "dual_cf"

    variant_info = extract_new_method_variant(run_name, method_name)
    if variant_info is not None:
        return variant_info.method_key
    return method_name


def extract_method_display(method_key: str) -> str:
    variant_info = variant_info_from_method_key(method_key)
    if variant_info is not None:
        return variant_info.display_name
    if method_key in METHOD_DISPLAY:
        return METHOD_DISPLAY[method_key]
    if method_key.startswith("dual_cf_"):
        suffix = method_key.removeprefix("dual_cf_")
        return f"DualCF({suffix})"
    return method_key


def extract_model_label(run_name: str) -> str:
    if run_name.startswith("duet_"):
        rest = run_name[len("duet_") :]
    elif run_name.startswith("rwku_"):
        rest = run_name[len("rwku_") :]
    else:
        return "unknown_model"

    for pattern in RUN_SPLIT_PATTERNS:
        match = re.search(pattern, rest)
        if match is not None:
            return rest[: match.start()]
    return "unknown_model"


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


def load_eval_sidecar(eval_dir: Path, run_dir: Path) -> dict[str, Any]:
    for config_path in (
        eval_dir / ".hydra" / "config.yaml",
        eval_dir.parent / ".hydra" / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
    ):
        if config_path.exists():
            data = load_yaml(config_path)
            if isinstance(data, dict):
                return data
    return {}


def resolve_run_dir_and_stage(eval_path: Path) -> tuple[Path, str]:
    for parent in eval_path.parents:
        if parent.parent.name == "checkpoint_evals":
            return parent.parent.parent.resolve(), parent.name
        if parent.name == "evals":
            if parent.parent.parent.name == "checkpoint_evals":
                return parent.parent.parent.parent.resolve(), parent.parent.name
            return parent.parent.resolve(), "final"
    raise ValueError(f"Could not resolve run directory for {eval_path}")


def infer_splits(run_name: str, benchmark: str, config: dict[str, Any]) -> tuple[str, str]:
    forget_split = str(config.get("forget_split") or "")
    holdout_split = str(config.get("holdout_split") or "")

    if "$" in forget_split:
        forget_split = ""
    if "$" in holdout_split:
        holdout_split = ""

    if not forget_split:
        if "_city_forget_rare_5_" in run_name:
            forget_split = "city_forget_rare_5"
        elif "_city_forget_popular_5_" in run_name:
            forget_split = "city_forget_popular_5"
        elif "_city_forget_5_" in run_name:
            forget_split = "city_forget_rare_5+city_forget_popular_5"
        else:
            level_match = re.search(r"_forget_level\d+_", run_name)
            if level_match is not None:
                forget_split = level_match.group(0).strip("_")

    if not holdout_split:
        if benchmark == "duet":
            holdout_split = "city_fast_retain_500"
        else:
            level_match = re.search(r"forget_level(\d+)", forget_split)
            if level_match is not None:
                holdout_split = f"neighbor_level{level_match.group(1)}"

    return (
        forget_split or "unknown_forget_split",
        holdout_split or "unknown_holdout_split",
    )


def build_eval_log(input_root: Path, eval_path: Path) -> EvalLog:
    run_dir, stage_label = resolve_run_dir_and_stage(eval_path)
    run_name = run_dir.name
    benchmark = "rwku" if run_name.startswith("rwku_") else "duet"
    config = load_eval_sidecar(eval_path.parent, run_dir)
    forget_split, holdout_split = infer_splits(run_name, benchmark, config)

    stage_index = {
        row["label"]: row
        for row in collect_eval_summaries(
            run_dir=run_dir,
            eval_root_name="checkpoint_evals",
            summary_name=DUET_SUMMARY_FILENAME,
        )
    }
    stage_info = stage_index.get(
        stage_label,
        {
            "step": parse_checkpoint_step(stage_label),
            "epoch": None,
        },
    )

    method_key = extract_method_key(run_name)
    return EvalLog(
        input_root=input_root.resolve(),
        input_root_label=infer_input_root_label(input_root.resolve()),
        eval_path=eval_path.resolve(),
        run_dir=run_dir,
        run_name=run_name,
        benchmark=benchmark,
        model_label=extract_model_label(run_name),
        forget_split=forget_split,
        holdout_split=holdout_split,
        lr=extract_lr(run_name),
        seed=extract_seed(run_name),
        method_key=method_key,
        method_display=extract_method_display(method_key),
        stage_label=stage_label,
        step=stage_info.get("step"),
        epoch=stage_info.get("epoch"),
    )


def method_sort_key(method_key: str) -> tuple[int, int, str]:
    variant_key = variant_sort_key(method_key)
    if variant_key is not None:
        return (len(METHOD_ORDER_INDEX), variant_key[0] * 100 + variant_key[1], method_key)
    return (METHOD_ORDER_INDEX.get(method_key, len(METHOD_ORDER_INDEX)), 0, method_key)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def dominant_ngram_count(tokens: list[str], n: int) -> int:
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return counts.most_common(1)[0][1]


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


def new_aggregate() -> dict[str, Any]:
    return {
        "total_rows": 0,
        "wrong_rows": 0,
        "reasons": Counter(),
    }


def update_aggregate(aggregate: dict[str, Any], reasons: list[str]) -> None:
    aggregate["total_rows"] += 1
    if reasons:
        aggregate["wrong_rows"] += 1
        aggregate["reasons"].update(reasons)


def aggregate_to_row(base: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row["total_rows"] = aggregate["total_rows"]
    row["wrong_rows"] = aggregate["wrong_rows"]
    row["clean_rows"] = aggregate["total_rows"] - aggregate["wrong_rows"]
    row["wrong_pct"] = format_pct(aggregate["wrong_rows"], aggregate["total_rows"])
    for reason in REASON_ORDER:
        row[f"{reason}_count"] = aggregate["reasons"].get(reason, 0)
    return row


def numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**12, value)


def main() -> None:
    args = parse_args()
    output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)

    eval_logs: list[EvalLog] = []
    for input_root in args.input_root:
        root = input_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            if root.name != args.eval_filename:
                raise ValueError(
                    f"Input file {root} does not match --eval-filename={args.eval_filename}."
                )
            eval_logs.append(build_eval_log(root.parent, root))
            continue
        for eval_path in sorted(root.rglob(args.eval_filename)):
            eval_logs.append(build_eval_log(root, eval_path))

    if not eval_logs:
        raise FileNotFoundError(
            f"No {args.eval_filename} files found under the provided input roots."
        )

    by_eval: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(new_aggregate)
    by_method_stage: dict[tuple[str, ...], dict[str, Any]] = defaultdict(new_aggregate)
    by_method_overall: dict[tuple[str, ...], dict[str, Any]] = defaultdict(new_aggregate)
    sampled_examples: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    checkpoint_log_count = 0

    for eval_log in eval_logs:
        if eval_log.stage_label != "final":
            checkpoint_log_count += 1

        payload = load_json(eval_log.eval_path)
        if not isinstance(payload, dict):
            continue

        for metric_name, metric_payload in payload.items():
            if not isinstance(metric_payload, dict):
                continue
            value_by_index = metric_payload.get("value_by_index", {})
            if not isinstance(value_by_index, dict):
                continue

            has_generation_rows = any(
                isinstance(row, dict) and "generation" in row
                for row in value_by_index.values()
            )
            if not has_generation_rows:
                continue

            eval_key = (
                eval_log.input_root_label,
                format_path(eval_log.eval_path),
                metric_name,
            )
            method_stage_key = (
                eval_log.input_root_label,
                eval_log.benchmark,
                eval_log.forget_split,
                eval_log.holdout_split,
                eval_log.lr,
                eval_log.stage_label,
                format_float(eval_log.epoch),
                str(eval_log.step or ""),
                eval_log.method_key,
                eval_log.method_display,
                metric_name,
            )
            method_overall_key = (
                eval_log.input_root_label,
                eval_log.benchmark,
                eval_log.method_key,
                eval_log.method_display,
            )

            for index_key, row in value_by_index.items():
                if not isinstance(row, dict):
                    continue
                generation = str(row.get("generation", "") or "")
                ground_truth = str(row.get("ground_truth", "") or "")
                reasons, features = classify_generation(generation, ground_truth, args)
                update_aggregate(by_eval[eval_key], reasons)
                update_aggregate(by_method_stage[method_stage_key], reasons)
                update_aggregate(by_method_overall[method_overall_key], reasons)

                sample_status = "wrong" if reasons else "clean"
                sample_key = (
                    eval_log.input_root_label,
                    eval_log.method_key,
                    metric_name,
                    sample_status,
                )
                if len(sampled_examples[sample_key]) < args.sample_limit:
                    sampled_examples[sample_key].append(
                        {
                            "status": sample_status,
                            "input_root_label": eval_log.input_root_label,
                            "benchmark": eval_log.benchmark,
                            "model_label": eval_log.model_label,
                            "forget_split": eval_log.forget_split,
                            "holdout_split": eval_log.holdout_split,
                            "lr": eval_log.lr,
                            "seed": eval_log.seed,
                            "stage_label": eval_log.stage_label,
                            "step": "" if eval_log.step is None else eval_log.step,
                            "epoch": format_float(eval_log.epoch),
                            "method_key": eval_log.method_key,
                            "method_display": eval_log.method_display,
                            "metric_name": metric_name,
                            "eval_path": format_path(eval_log.eval_path),
                            "run_name": eval_log.run_name,
                            "index": index_key,
                            "reasons": ",".join(reasons),
                            "question": extract_question(str(row.get("input", "") or "")),
                            "ground_truth": ground_truth,
                            "generation": generation,
                            "word_count": features["word_count"],
                            "char_count": features["char_count"],
                            "punctuation_ratio": f"{features['punctuation_ratio']:.4f}",
                            "bigram_repeat": features["bigram_repeat"],
                            "trigram_repeat": features["trigram_repeat"],
                            "longest_char_run": features["longest_char_run"],
                            "longest_token_run": features["longest_token_run"],
                            "dominant_token_ratio": f"{features['dominant_token_ratio']:.4f}",
                            "unique_token_ratio": f"{features['unique_token_ratio']:.4f}",
                            "too_long_limit": features["too_long_limit"],
                        }
                    )

    eval_summary_rows: list[dict[str, Any]] = []
    for eval_log in sorted(
        eval_logs,
        key=lambda item: (
            item.input_root_label,
            item.benchmark,
            item.forget_split,
            item.lr,
            method_sort_key(item.method_key),
            item.stage_label,
            format_path(item.eval_path),
        ),
    ):
        for metric_name in ("forget_qa_rouge", "holdout_qa_rouge"):
            eval_key = (
                eval_log.input_root_label,
                format_path(eval_log.eval_path),
                metric_name,
            )
            aggregate = by_eval.get(eval_key)
            if aggregate is None:
                continue
            eval_summary_rows.append(
                aggregate_to_row(
                    {
                        "input_root_label": eval_log.input_root_label,
                        "benchmark": eval_log.benchmark,
                        "model_label": eval_log.model_label,
                        "forget_split": eval_log.forget_split,
                        "holdout_split": eval_log.holdout_split,
                        "lr": eval_log.lr,
                        "seed": eval_log.seed,
                        "stage_label": eval_log.stage_label,
                        "step": "" if eval_log.step is None else eval_log.step,
                        "epoch": format_float(eval_log.epoch),
                        "method_key": eval_log.method_key,
                        "method_display": eval_log.method_display,
                        "metric_name": metric_name,
                        "eval_path": format_path(eval_log.eval_path),
                        "run_name": eval_log.run_name,
                    },
                    aggregate,
                )
            )

    method_stage_rows: list[dict[str, Any]] = []
    for key, aggregate in sorted(
        by_method_stage.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][4],
            float(item[0][6] or 10**9),
            method_sort_key(item[0][8]),
            item[0][10],
        ),
    ):
        (
            input_root_label,
            benchmark,
            forget_split,
            holdout_split,
            lr,
            stage_label,
            epoch,
            step,
            method_key,
            method_display,
            metric_name,
        ) = key
        method_stage_rows.append(
            aggregate_to_row(
                {
                    "input_root_label": input_root_label,
                    "benchmark": benchmark,
                    "forget_split": forget_split,
                    "holdout_split": holdout_split,
                    "lr": lr,
                    "stage_label": stage_label,
                    "step": step,
                    "epoch": epoch,
                    "method_key": method_key,
                    "method_display": method_display,
                    "metric_name": metric_name,
                },
                aggregate,
            )
        )

    method_overall_rows: list[dict[str, Any]] = []
    for key, aggregate in sorted(
        by_method_overall.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            method_sort_key(item[0][2]),
        ),
    ):
        input_root_label, benchmark, method_key, method_display = key
        method_overall_rows.append(
            aggregate_to_row(
                {
                    "input_root_label": input_root_label,
                    "benchmark": benchmark,
                    "method_key": method_key,
                    "method_display": method_display,
                },
                aggregate,
            )
        )

    sample_rows: list[dict[str, Any]] = []
    for key in sorted(
        sampled_examples,
        key=lambda item: (
            item[0],
            method_sort_key(item[1]),
            item[2],
            item[3],
        ),
    ):
        rows = sampled_examples[key]
        rows.sort(key=lambda row: numeric_sort_key(str(row["index"])))
        sample_rows.extend(rows)

    summary_fieldnames = [
        "input_root_label",
        "benchmark",
        "method_key",
        "method_display",
        "total_rows",
        "wrong_rows",
        "clean_rows",
        "wrong_pct",
        *[f"{reason}_count" for reason in REASON_ORDER],
    ]
    method_stage_fieldnames = [
        "input_root_label",
        "benchmark",
        "forget_split",
        "holdout_split",
        "lr",
        "stage_label",
        "step",
        "epoch",
        "method_key",
        "method_display",
        "metric_name",
        "total_rows",
        "wrong_rows",
        "clean_rows",
        "wrong_pct",
        *[f"{reason}_count" for reason in REASON_ORDER],
    ]
    eval_fieldnames = [
        "input_root_label",
        "benchmark",
        "model_label",
        "forget_split",
        "holdout_split",
        "lr",
        "seed",
        "stage_label",
        "step",
        "epoch",
        "method_key",
        "method_display",
        "metric_name",
        "eval_path",
        "run_name",
        "total_rows",
        "wrong_rows",
        "clean_rows",
        "wrong_pct",
        *[f"{reason}_count" for reason in REASON_ORDER],
    ]
    sample_fieldnames = [
        "status",
        "input_root_label",
        "benchmark",
        "model_label",
        "forget_split",
        "holdout_split",
        "lr",
        "seed",
        "stage_label",
        "step",
        "epoch",
        "method_key",
        "method_display",
        "metric_name",
        "eval_path",
        "run_name",
        "index",
        "reasons",
        "question",
        "ground_truth",
        "generation",
        "word_count",
        "char_count",
        "punctuation_ratio",
        "bigram_repeat",
        "trigram_repeat",
        "longest_char_run",
        "longest_token_run",
        "dominant_token_ratio",
        "unique_token_ratio",
        "too_long_limit",
    ]

    write_tsv(output_root / "overall_method_summary.tsv", method_overall_rows, summary_fieldnames)
    write_tsv(output_root / "method_stage_summary.tsv", method_stage_rows, method_stage_fieldnames)
    write_tsv(output_root / "eval_file_summary.tsv", eval_summary_rows, eval_fieldnames)
    write_tsv(output_root / "sample_examples.tsv", sample_rows, sample_fieldnames)

    print(f"Wrote wrong-generation reports to {output_root}")
    print()
    print("Overall method summary:")
    for row in method_overall_rows:
        print(
            f"  {row['input_root_label']:>12s} | {row['benchmark']:>4s} | "
            f"{row['method_display']:<14s} | wrong={str(row['wrong_rows']):>6s}/{str(row['total_rows']):>6s} "
            f"({str(row['wrong_pct']):>6s}%)"
        )
    if checkpoint_log_count == 0:
        print()
        print(
            "Note: no checkpoint-level generation logs matched the requested eval filename. "
            "These reports cover final eval logs only."
        )


if __name__ == "__main__":
    main()
