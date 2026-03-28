#!/usr/bin/env python3
"""Build text sanity-check reports from saved endpoint eval logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DUET_EVAL_FILENAME = "DUET_EVAL.json"
DUET_SUMMARY_FILENAME = "DUET_SUMMARY.json"
COS_SIM_EVAL_FILENAME = "COS_SIM_EVAL.json"
LR_RE = re.compile(r"_lr([^_]+)")
METHOD_RE = re.compile(
    r"_(dual_cf|dpo_cf|ga|ada_pop|loku|npo_sam|npo|simnpo|simple_ce|multicf|boundary_cf|span_cf|falcon)_lora_.*?_lr[^_]+(.*)$"
)
DUAL_FLAG_RE = re.compile(r"^(dOn|dOff|aOn|aOff|adT|adF)$")
RUN_SPLIT_PATTERNS = [
    r"_city_forget_rare_5_",
    r"_city_forget_popular_5_",
    r"_city_forget_5_",
    r"_forget_level\d+_",
]
QUESTION_RE = re.compile(r"\buser\b\s*\n\n(?P<question>.*?)\bassistant\b\s*\n*\Z", re.DOTALL)
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_METRICS = (
    ("forget_qa_rouge", "forget_qa_cos_sim", "forget"),
    ("holdout_qa_rouge", "holdout_qa_cos_sim", "holdout"),
)
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
    "multicf": "MultiCF",
    "boundary_cf": "BoundaryCF",
    "span_cf": "SpanCF",
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
    "multicf",
    "boundary_cf",
    "span_cf",
    "falcon",
]
METHOD_ORDER_INDEX = {name: index for index, name in enumerate(METHOD_ORDER)}


@dataclass(frozen=True)
class EvalRun:
    input_root: Path
    input_root_label: str
    run_dir: Path
    eval_dir: Path
    run_name: str
    benchmark: str
    model_label: str
    forget_split: str
    holdout_split: str
    lr: str
    method_key: str
    method_display: str


@dataclass(frozen=True)
class MissingEval:
    input_root: Path
    run_dir: Path
    eval_dir: Path
    reason: str


@dataclass(frozen=True)
class GroupKey:
    benchmark: str
    model_label: str
    forget_split: str
    holdout_split: str
    lr: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Root to scan. Can point at a saves folder, a campaign folder, or any parent directory.",
    )
    parser.add_argument(
        "--lr",
        action="append",
        default=[],
        help="Limit reports to one or more learning rates, for example 1e-4 or 5e-5.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where report text files and coverage tables will be written.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="How many examples to sample from each forget and holdout split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Base seed for deterministic example selection.",
    )
    parser.add_argument(
        "--sbert-model-path",
        type=str,
        default=os.environ.get("SBERT_MODEL_PATH", DEFAULT_SBERT_MODEL),
        help="Local path or HF repo id for cosine similarity when COS_SIM_EVAL.json is missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing reports.",
    )
    return parser.parse_args(argv)


def prepare_output_root(path: Path, overwrite: bool) -> Path:
    root = path.expanduser().resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {root}. Pass --overwrite to rebuild it."
            )
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        root.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    slug = slug.strip("-")
    return slug or "unknown"


def format_run_path(path: Path) -> str:
    cwd = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(cwd))
    except ValueError:
        return str(path.resolve())


def extract_lr(run_name: str) -> str:
    match = LR_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse lr from run name: {run_name}")
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

    return method_name


def extract_method_display(method_key: str) -> str:
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


def infer_input_root_label(root: Path) -> str:
    if root.name:
        return root.name
    return slugify(str(root))


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


def numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**12, value)


def choose_indices(indices: list[str], count: int, seed_key: str) -> list[str]:
    if len(indices) <= count:
        return sorted(indices, key=numeric_sort_key)
    rng = random.Random(seed_key)
    sampled = rng.sample(indices, count)
    return sorted(sampled, key=numeric_sort_key)


def load_eval_sidecar(eval_dir: Path) -> dict[str, Any]:
    for config_path in (
        eval_dir / ".hydra" / "config.yaml",
        eval_dir.parent / ".hydra" / "config.yaml",
    ):
        if config_path.exists():
            data = load_yaml(config_path)
            if isinstance(data, dict):
                return data
    return {}


def create_run_record(input_root: Path, eval_path: Path) -> EvalRun:
    eval_dir = eval_path.parent
    run_dir = eval_dir.parent
    run_name = run_dir.name
    eval_cfg = load_eval_sidecar(eval_dir)

    benchmark = "duet"
    if run_name.startswith("rwku_"):
        benchmark = "rwku"
    elif run_name.startswith("duet_"):
        benchmark = "duet"

    forget_split = str(eval_cfg.get("forget_split") or "")
    holdout_split = str(eval_cfg.get("holdout_split") or "")
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

    method_key = extract_method_key(run_name)
    return EvalRun(
        input_root=input_root.resolve(),
        input_root_label=infer_input_root_label(input_root.resolve()),
        run_dir=run_dir.resolve(),
        eval_dir=eval_dir.resolve(),
        run_name=run_name,
        benchmark=benchmark,
        model_label=extract_model_label(run_name),
        forget_split=forget_split or "unknown_forget_split",
        holdout_split=holdout_split or "unknown_holdout_split",
        lr=extract_lr(run_name),
        method_key=method_key,
        method_display=extract_method_display(method_key),
    )


def discover_runs(input_root: Path) -> tuple[list[EvalRun], list[MissingEval]]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    runs: list[EvalRun] = []
    missing: list[MissingEval] = []
    eval_dirs_with_full = set()

    for eval_path in sorted(root.rglob(f"evals/{DUET_EVAL_FILENAME}")):
        try:
            runs.append(create_run_record(root, eval_path))
            eval_dirs_with_full.add(eval_path.parent.resolve())
        except ValueError as exc:
            missing.append(
                MissingEval(
                    input_root=root,
                    run_dir=eval_path.parent.parent.resolve(),
                    eval_dir=eval_path.parent.resolve(),
                    reason=str(exc),
                )
            )

    for summary_path in sorted(root.rglob(f"evals/{DUET_SUMMARY_FILENAME}")):
        eval_dir = summary_path.parent.resolve()
        if eval_dir in eval_dirs_with_full:
            continue
        missing.append(
            MissingEval(
                input_root=root,
                run_dir=eval_dir.parent.resolve(),
                eval_dir=eval_dir,
                reason=(
                    "Found DUET_SUMMARY.json but no DUET_EVAL.json. "
                    "This save root only preserved summaries, so per-sample text export is not possible."
                ),
            )
        )

    return runs, missing


def method_sort_key(record: EvalRun) -> tuple[int, str, str]:
    return (
        METHOD_ORDER_INDEX.get(record.method_key, len(METHOD_ORDER_INDEX) + 1),
        record.method_display,
        record.input_root_label,
    )


class CosineScorer:
    def __init__(self, model_ref: str):
        self.model_ref = model_ref
        self._model = None

    @staticmethod
    def _candidate_snapshot_dir(model_id: str, cache_root: Path) -> Path | None:
        repo_dir = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if not snapshots:
            return None
        return snapshots[-1]

    def _resolve_model_path(self) -> str:
        candidate = Path(self.model_ref).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        ).expanduser()
        snapshot_dir = self._candidate_snapshot_dir(self.model_ref, hf_home)
        if snapshot_dir is not None:
            return str(snapshot_dir.resolve())
        raise FileNotFoundError(
            f"Could not resolve SBERT model '{self.model_ref}'. "
            "Pass --sbert-model-path to a local path or cache the model in HF_HOME."
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._resolve_model_path(), device="cpu")
        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load_model()
        ground_truths = [gt for gt, _ in pairs]
        generations = [gen for _, gen in pairs]
        gt_embs = model.encode(ground_truths, normalize_embeddings=True)
        gen_embs = model.encode(generations, normalize_embeddings=True)
        return (gt_embs * gen_embs).sum(axis=1).tolist()


def get_selected_cos_values(
    eval_payload: dict[str, Any],
    cos_payload: dict[str, Any] | None,
    rouge_metric: str,
    cos_metric: str,
    selected_indices: list[str],
    cosine_scorer: CosineScorer | None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    if cos_payload is not None and cos_metric in cos_payload:
        block = cos_payload[cos_metric].get("value_by_index", {})
        for idx in selected_indices:
            item = block.get(str(idx))
            if isinstance(item, dict):
                output[str(idx)] = item
        if len(output) == len(selected_indices):
            return output

    if cosine_scorer is None:
        return output

    rouge_block = eval_payload.get(rouge_metric, {}).get("value_by_index", {})
    pairs: list[tuple[str, str]] = []
    valid_indices: list[str] = []
    for idx in selected_indices:
        item = rouge_block.get(str(idx))
        if not isinstance(item, dict):
            continue
        gt = item.get("ground_truth")
        generation = item.get("generation")
        if not gt or not generation:
            continue
        valid_indices.append(str(idx))
        pairs.append((str(gt), str(generation)))

    if not pairs:
        return output

    sims = cosine_scorer.score_pairs(pairs)
    for idx, sim, pair in zip(valid_indices, sims, pairs):
        output[idx] = {
            "cos_sim": float(sim),
            "ground_truth": pair[0],
            "generation": pair[1],
        }
    return output


def read_cos_payload(eval_dir: Path) -> dict[str, Any] | None:
    cos_path = eval_dir / COS_SIM_EVAL_FILENAME
    if cos_path.exists():
        payload = load_json(cos_path)
        if isinstance(payload, dict):
            return payload
    return None


def make_group_display_label(group: GroupKey) -> str:
    return (
        f"{group.benchmark.upper()} | {group.model_label} | "
        f"{group.forget_split} | lr={group.lr}"
    )


def write_missing_table(path: Path, missing: list[MissingEval]) -> None:
    rows = [
        {
            "input_root": str(entry.input_root),
            "run_dir": str(entry.run_dir),
            "eval_dir": str(entry.eval_dir),
            "reason": entry.reason,
        }
        for entry in missing
    ]
    write_tsv(path, rows, ["input_root", "run_dir", "eval_dir", "reason"])


def write_run_table(path: Path, runs: list[EvalRun]) -> None:
    rows = [
        {
            "input_root": str(run.input_root),
            "input_root_label": run.input_root_label,
            "run_dir": str(run.run_dir),
            "eval_dir": str(run.eval_dir),
            "benchmark": run.benchmark,
            "model_label": run.model_label,
            "forget_split": run.forget_split,
            "holdout_split": run.holdout_split,
            "lr": run.lr,
            "method_key": run.method_key,
            "method_display": run.method_display,
            "has_cos_sim_eval": str((run.eval_dir / COS_SIM_EVAL_FILENAME).exists()),
        }
        for run in runs
    ]
    write_tsv(
        path,
        rows,
        [
            "input_root",
            "input_root_label",
            "run_dir",
            "eval_dir",
            "benchmark",
            "model_label",
            "forget_split",
            "holdout_split",
            "lr",
            "method_key",
            "method_display",
            "has_cos_sim_eval",
        ],
    )


def build_method_labels(runs: list[EvalRun]) -> dict[Path, str]:
    counts = Counter((run.method_display for run in runs))
    labels: dict[Path, str] = {}
    for run in runs:
        label = run.method_display
        if counts[label] > 1:
            label = f"{label} [{run.input_root_label}]"
        labels[run.run_dir] = label
    return labels


def render_report(
    group: GroupKey,
    runs: list[EvalRun],
    sample_count: int,
    seed: int,
    cosine_scorer: CosineScorer | None,
) -> str:
    runs = sorted(runs, key=method_sort_key)
    method_labels = build_method_labels(runs)
    eval_payloads = {run.run_dir: load_json(run.eval_dir / DUET_EVAL_FILENAME) for run in runs}
    cos_payloads = {run.run_dir: read_cos_payload(run.eval_dir) for run in runs}

    reference_run = runs[0]
    reference_payload = eval_payloads[reference_run.run_dir]

    lines: list[str] = []
    lines.append(f"Benchmark: {group.benchmark.upper()}")
    lines.append(f"Model: {group.model_label}")
    lines.append(f"Forget split: {group.forget_split}")
    lines.append(f"Holdout split: {group.holdout_split}")
    lines.append(f"Learning rate: {group.lr}")
    lines.append(f"Runs compared: {', '.join(method_labels[run.run_dir] for run in runs)}")
    lines.append("")

    for rouge_metric, cos_metric, section_name in REPORT_METRICS:
        block = reference_payload.get(rouge_metric, {}).get("value_by_index", {})
        available_indices = sorted(block.keys(), key=numeric_sort_key)
        selected_indices = choose_indices(
            available_indices,
            sample_count,
            seed_key=f"{seed}:{group}:{section_name}",
        )
        label = "FORGET" if section_name == "forget" else "HOLDOUT"
        goal = "forget this answer" if section_name == "forget" else "keep this answer"
        lines.append(f"{label} EXAMPLES")
        lines.append(
            f"Selected {len(selected_indices)} / {len(available_indices)} examples. Goal: {goal}."
        )
        lines.append("")

        for position, idx in enumerate(selected_indices, start=1):
            reference_item = block.get(str(idx), {})
            question = extract_question(str(reference_item.get("input", "")))
            target_answer = str(reference_item.get("ground_truth", ""))
            lines.append(f"{label} SAMPLE {position} | index={idx}")
            lines.append(f"Question: {question}")
            if section_name == "forget":
                lines.append(f"Answer to forget: {target_answer}")
            else:
                lines.append(f"Answer to keep: {target_answer}")

            for run in runs:
                payload = eval_payloads[run.run_dir]
                item = payload.get(rouge_metric, {}).get("value_by_index", {}).get(str(idx), {})
                cos_values = get_selected_cos_values(
                    eval_payload=payload,
                    cos_payload=cos_payloads[run.run_dir],
                    rouge_metric=rouge_metric,
                    cos_metric=cos_metric,
                    selected_indices=[str(idx)],
                    cosine_scorer=cosine_scorer,
                )
                cos_item = cos_values.get(str(idx), {})
                generation = str(item.get("generation", "")).strip()
                lines.append(
                    f"{method_labels[run.run_dir]} | rougeL_recall={format_float(item.get('rougeL_recall'))} "
                    f"| cos_sim={format_float(cos_item.get('cos_sim'))}"
                )
                lines.append(f"Save path: {format_run_path(run.run_dir)}")
                lines.append(f"Answer: {generation}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)

    all_runs: list[EvalRun] = []
    all_missing: list[MissingEval] = []
    for input_root in args.input_root:
        runs, missing = discover_runs(input_root)
        all_runs.extend(runs)
        all_missing.extend(missing)

    if args.lr:
        allowed_lrs = set(args.lr)
        all_runs = [run for run in all_runs if run.lr in allowed_lrs]
        all_missing = [
            entry
            for entry in all_missing
            if any(lr in str(entry.run_dir) for lr in allowed_lrs) or not allowed_lrs
        ]

    all_runs = sorted(all_runs, key=lambda run: (run.benchmark, run.model_label, run.forget_split, run.lr, method_sort_key(run)))
    write_run_table(output_root / "matched_runs.tsv", all_runs)
    write_missing_table(output_root / "missing_sample_logs.tsv", all_missing)

    if not all_runs:
        raise RuntimeError(
            "No usable DUET_EVAL.json files matched the requested roots and LR filters. "
            "Check missing_sample_logs.tsv for roots that only preserved summaries."
        )

    grouped: dict[GroupKey, list[EvalRun]] = defaultdict(list)
    for run in all_runs:
        grouped[
            GroupKey(
                benchmark=run.benchmark,
                model_label=run.model_label,
                forget_split=run.forget_split,
                holdout_split=run.holdout_split,
                lr=run.lr,
            )
        ].append(run)

    need_cosine = any(not (run.eval_dir / COS_SIM_EVAL_FILENAME).exists() for run in all_runs)
    cosine_scorer = CosineScorer(args.sbert_model_path) if need_cosine else None

    index_rows: list[dict[str, Any]] = []
    for group in sorted(
        grouped,
        key=lambda key: (key.benchmark, key.model_label, key.forget_split, key.lr),
    ):
        runs = grouped[group]
        report_name = (
            f"{slugify(group.benchmark)}__{slugify(group.model_label)}__"
            f"{slugify(group.forget_split)}__lr-{slugify(group.lr)}.txt"
        )
        report_path = output_root / report_name
        report_text = render_report(
            group=group,
            runs=runs,
            sample_count=args.sample_count,
            seed=args.seed,
            cosine_scorer=cosine_scorer,
        )
        report_path.write_text(report_text, encoding="utf-8")
        index_rows.append(
            {
                "benchmark": group.benchmark,
                "model_label": group.model_label,
                "forget_split": group.forget_split,
                "holdout_split": group.holdout_split,
                "lr": group.lr,
                "report_path": str(report_path),
                "num_runs": str(len(runs)),
                "methods": ",".join(build_method_labels(sorted(runs, key=method_sort_key)).values()),
                "group_label": make_group_display_label(group),
            }
        )

    write_tsv(
        output_root / "report_index.tsv",
        index_rows,
        [
            "benchmark",
            "model_label",
            "forget_split",
            "holdout_split",
            "lr",
            "report_path",
            "num_runs",
            "methods",
            "group_label",
        ],
    )

    dump_json(
        output_root / "summary.json",
        {
            "reports_written": len(index_rows),
            "usable_runs": len(all_runs),
            "missing_eval_dirs": len(all_missing),
            "input_roots": [str(Path(root).expanduser().resolve()) for root in args.input_root],
            "lr_filter": args.lr,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
