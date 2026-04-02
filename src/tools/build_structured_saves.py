#!/usr/bin/env python3
"""Build comparison tables from packaged unlearning save summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from checkpoint_summary_utils import checkpoint_sort_key, collect_eval_summaries
from new_method_variant_utils import base_variant_algorithm, extract_new_method_variant, variant_sort_key


META_COLUMNS = {"label", "checkpoint", "step", "epoch"}
PREFERRED_METRIC_ORDER = [
    "forget_qa_rouge",
    "holdout_qa_rouge",
    "forget_wrong_gen_rate",
    "holdout_wrong_gen_rate",
    "forget_qa_cos_sim",
    "holdout_qa_cos_sim",
    "utility_avg",
    "utility_delta_vs_base",
]
METHOD_ORDER = [
    "full",
    "d_only",
    "a_only",
    "dpo",
    "simple_ce",
    "multicf",
    "boundary_cf",
    "span_cf",
    "span_cf_samnpo",
    "span_cf_simnpo",
    "span_cf_local_retain",
    "span_cf_simnpo_local_retain",
    "span_cf_simnpo_sam",
    "span_cf_simnpo_projected",
    "ga",
    "ada_pop",
    "npo",
    "simnpo",
    "npo_sam",
    "loku",
]
METHOD_ORDER_INDEX = {name: index for index, name in enumerate(METHOD_ORDER)}
LR_RE = re.compile(r"_lr([^_]+)")
METHOD_RE = re.compile(
    r"_(dual_cf|dpo_cf|simple_ce|multicf|boundary_cf|span_cf_simnpo_local_retain|span_cf_simnpo_projected|span_cf_simnpo_sam|span_cf_samnpo|span_cf_local_retain|span_cf_simnpo|span_cf|ga|ada_pop|npo|simnpo|npo_sam|loku)_lora_.*?_lr[^_]+(.*)$"
)
DUAL_FLAG_RE = re.compile(r"^(dOn|dOff|aOn|aOff|adT|adF)$")
SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
UTILITY_BENCHMARK_ORDER = {
    "mmlu_pro": 0,
    "truthfulqa_bin": 1,
    "arc": 2,
    "winogrande": 3,
}
UTILITY_COLUMN_RE = re.compile(r"^(mmlu_pro|truthfulqa_bin|arc|winogrande)_\d+_acc$")
UTILITY_TASK_RE = re.compile(r"^(utility_(mmlu_pro|truthfulqa_bin|arc|winogrande)_\d+)/acc$")
UTILITY_TASK_COUNT_RE = re.compile(r"_(\d+)$")


class NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to saves-clean/unlearn, saves-clean, or a parent directory that contains unlearn/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where structured-saves outputs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before generating files.",
    )
    parser.add_argument(
        "--average-seeds",
        action="store_true",
        help=(
            "Average rows for runs that share the same canonical run name after stripping a trailing "
            "_seed<INT> suffix."
        ),
    )
    return parser.parse_args()


def resolve_unlearn_root(input_root: Path) -> Path:
    root = input_root.expanduser().resolve()
    if root.name == "unlearn" and root.is_dir():
        return root
    candidate = root / "unlearn"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve an unlearn root under {root}")


def prepare_output_root(output_root: Path, overwrite: bool) -> Path:
    root = output_root.expanduser().resolve()
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {root}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float(value: str | None) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def coalesce(*values: Any) -> Any:
    for value in values:
        if value not in {None, "", "None"}:
            return value
    return None


def load_tsv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, raw_value in row.items():
                if key == "step":
                    parsed[key] = parse_int(raw_value)
                elif key == "epoch":
                    parsed[key] = parse_float(raw_value)
                elif key in {"label", "checkpoint"}:
                    parsed[key] = raw_value
                else:
                    parsed[key] = parse_float(raw_value)
            rows.append(parsed)
    return rows


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def extract_lr(run_name: str) -> str:
    match = LR_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse lr from run name: {run_name}")
    return match.group(1)


def split_seed_suffix(run_name: str) -> tuple[str, str | None]:
    match = SEED_SUFFIX_RE.fullmatch(run_name)
    if match is None:
        return run_name, None
    return match.group("base"), match.group("seed")


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
            return "full"
        if flag_set == {"dOn", "aOff"}:
            return "d_only"
        if flag_set == {"dOff", "aOn"}:
            return "a_only"
        return "_".join([method_name] + flags) if flags else method_name

    if method_name == "dpo_cf":
        return "dpo"
    if method_name == "simnpo":
        return "simnpo"
    if method_name == "npo_sam":
        return "npo_sam"
    if method_name == "simple_ce":
        ablation_tokens = [token for token in suffix.split("_") if token.startswith(("cf", "ret", "gamma"))]
        if ablation_tokens:
            return "_".join([method_name] + ablation_tokens)
        return method_name
    variant_info = extract_new_method_variant(run_name, method_name)
    if variant_info is not None:
        return variant_info.method_key
    return method_name


def infer_split_bucket(run_name: str, config: dict[str, Any]) -> str:
    if run_name.startswith("rwku_"):
        return "rwku"

    forget_split = str(config.get("forget_split", ""))
    if forget_split == "city_forget_rare_5":
        return "duet_rare"
    if forget_split == "city_forget_popular_5":
        return "duet_popular"
    if forget_split == "city_forget_5":
        return "duet_merged"

    if "_city_forget_rare_5_" in run_name:
        return "duet_rare"
    if "_city_forget_popular_5_" in run_name:
        return "duet_popular"
    if "_city_forget_5_" in run_name:
        return "duet_merged"
    raise ValueError(f"Could not infer split bucket for run: {run_name}")


def extract_override_map(overrides: list[Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in overrides:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        output[key.lstrip("+")] = value
    return output


def resolve_simple_reference(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return context.get(key, value)
    return value


def dedupe_representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_row = next((row for row in rows if row.get("label") == "base_model_orig"), None)
    trajectory_rows = [
        row
        for row in rows
        if row.get("label") not in {"base_model_orig", "base_model_run"}
    ]

    representatives: list[dict[str, Any]] = []
    for row in trajectory_rows:
        if representatives:
            previous = representatives[-1]
            if previous.get("step") == row.get("step") and previous.get("epoch") == row.get("epoch"):
                if row.get("label") == "final":
                    representatives[-1] = row
                continue
        representatives.append(row)

    if base_row is not None:
        return [base_row] + representatives
    return representatives


def normalize_epoch_slot_value(actual_epoch: Any) -> float | None:
    if actual_epoch in {None, "", "None"}:
        return None
    return math.floor(float(actual_epoch) * 2.0 + 0.5) / 2.0


def build_epoch_slots(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    slot_rows: list[dict[str, Any]] = []
    slot_names: list[str] = []

    representatives = dedupe_representative_rows(rows)
    last_slot_value = 0.0
    for row in representatives:
        label = row.get("label")
        if label == "base_model_orig":
            slot_value = 0.0
        else:
            slot_value = normalize_epoch_slot_value(row.get("epoch"))
            if slot_value is None:
                slot_value = last_slot_value + 0.5
        slot_name = format(slot_value, ".1f")
        slot_names.append(slot_name)
        slot_rows.append(
            {
                "slot": slot_name,
                "label": label,
                "checkpoint": row.get("checkpoint"),
                "step": row.get("step"),
                "actual_epoch": row.get("epoch"),
            }
        )
        last_slot_value = slot_value
    return slot_rows, slot_names


def build_metric_values_by_slot(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    slot_rows, _ = build_epoch_slots(rows)
    representative_rows = dedupe_representative_rows(rows)
    return {
        slot_row["slot"]: source_row.get(metric_name)
        for slot_row, source_row in zip(slot_rows, representative_rows)
    }


def average_optional_floats(values: list[Any]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def average_scalar_values(values: list[Any]) -> Any:
    present_values = [value for value in values if value not in {None, "", "None"}]
    if not present_values:
        return None
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present_values):
        return sum(float(value) for value in present_values) / len(present_values)
    first_value = present_values[0]
    if all(value == first_value for value in present_values):
        return first_value
    return first_value


def collect_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    metric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if key not in META_COLUMNS and value is not None
    }
    ordered = [metric for metric in PREFERRED_METRIC_ORDER if metric in metric_keys]
    utility_metric_keys = sorted(
        (metric for metric in metric_keys if UTILITY_COLUMN_RE.fullmatch(metric)),
        key=utility_metric_sort_key,
    )
    ordered.extend(metric for metric in utility_metric_keys if metric not in ordered)
    ordered.extend(sorted(metric_keys - set(ordered)))
    return ordered


def method_sort_key(method_name: str) -> tuple[int, int, str]:
    variant_key = variant_sort_key(method_name)
    if variant_key is not None:
        base_method = base_variant_algorithm(method_name)
        if base_method is not None:
            return (METHOD_ORDER_INDEX.get(base_method, len(METHOD_ORDER)), variant_key[1], method_name)
    return (METHOD_ORDER_INDEX.get(method_name, len(METHOD_ORDER)), 0, method_name)


def lr_sort_key(lr: str) -> float:
    return float(lr)


def collect_cosine_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals",
        summary_name="COS_SIM_SUMMARY.json",
    ):
        metrics = load_json(Path(row["summary_path"]))
        merged_row = {
            "label": row["label"],
            "checkpoint": row["label"],
            "step": row["step"],
            "epoch": row["epoch"],
        }
        for key, value in metrics.items():
            merged_row[key] = float(value) if value is not None else None
        rows.append(merged_row)
    return rows


def collect_wrong_generation_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals",
        summary_name="WRONG_GENERATIONS_SUMMARY.json",
    ):
        metrics = load_json(Path(row["summary_path"]))
        rows.append(
            {
                "label": row["label"],
                "checkpoint": row["label"],
                "step": row["step"],
                "epoch": row["epoch"],
                "forget_wrong_gen_rate": parse_summary_metric(metrics, "forget_wrong_gen_rate"),
                "holdout_wrong_gen_rate": parse_summary_metric(metrics, "holdout_wrong_gen_rate"),
            }
        )
    return rows


def utility_metric_sort_key(metric_name: str) -> tuple[int, str]:
    match = UTILITY_COLUMN_RE.fullmatch(metric_name)
    if match is None:
        return (len(UTILITY_BENCHMARK_ORDER), metric_name)
    return (UTILITY_BENCHMARK_ORDER[match.group(1)], metric_name)


def utility_task_weight(task_name: str) -> int:
    match = UTILITY_TASK_COUNT_RE.search(task_name)
    if match is None:
        raise ValueError(f"Could not infer sample count from task name: {task_name}")
    return int(match.group(1))


def parse_summary_metric(summary: dict[str, Any], metric_name: str) -> float | None:
    value = summary.get(metric_name)
    if value is None:
        return None
    return float(value)


def parse_utility_metric(summary: dict[str, Any], task_name: str) -> float | None:
    value = summary.get(f"{task_name}/acc")
    if value is None:
        return None
    return float(value)


def discover_utility_task_specs(summary: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for metric_name in summary:
        match = UTILITY_TASK_RE.fullmatch(metric_name)
        if match is None:
            continue

        task_name = match.group(1)
        specs.append(
            {
                "task_name": task_name,
                "benchmark": match.group(2),
                "column_name": f"{task_name.removeprefix('utility_')}_acc",
            }
        )

    specs.sort(key=lambda spec: (UTILITY_BENCHMARK_ORDER[spec["benchmark"]], spec["task_name"]))
    return specs


def collect_checkpoint_rows_from_json(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals",
        summary_name="DUET_SUMMARY.json",
    ):
        metrics = load_json(Path(row["summary_path"]))
        rows.append(
            {
                "label": row["label"],
                "checkpoint": row["label"],
                "step": row["step"],
                "epoch": row["epoch"],
                "forget_qa_rouge": parse_summary_metric(metrics, "forget_qa_rouge"),
                "holdout_qa_rouge": parse_summary_metric(metrics, "holdout_qa_rouge"),
            }
        )
    return rows


def collect_utility_rows_from_json(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_utility: float | None = None

    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals_utility",
        summary_name="LMEval_SUMMARY.json",
    ):
        metrics = load_json(Path(row["summary_path"]))
        task_specs = discover_utility_task_specs(metrics)
        output_row = {
            "label": row["label"],
            "checkpoint": row["label"],
            "step": row["step"],
            "epoch": row["epoch"],
        }
        weighted_sum = 0.0
        total_weight = 0

        for spec in task_specs:
            metric_value = parse_utility_metric(metrics, spec["task_name"])
            output_row[spec["column_name"]] = metric_value
            if metric_value is None:
                continue
            weight = utility_task_weight(spec["task_name"])
            weighted_sum += metric_value * weight
            total_weight += weight

        utility_avg = weighted_sum / total_weight if total_weight else None
        output_row["utility_avg"] = utility_avg

        if row["label"] == "base_model_orig":
            base_utility = utility_avg
        output_row["utility_delta_vs_base"] = (
            utility_avg - base_utility
            if utility_avg is not None and base_utility is not None
            else None
        )
        rows.append(output_row)

    return rows


def collect_merged_rows(run_dir: Path) -> list[dict[str, Any]]:
    merged_summary_path = run_dir / "checkpoint_evals_merged" / "summary.tsv"
    if merged_summary_path.exists():
        merged_rows = load_tsv_rows(merged_summary_path)
    else:
        checkpoint_rows = collect_checkpoint_rows_from_json(run_dir)
        utility_rows = collect_utility_rows_from_json(run_dir)
        utility_metric_names = [
            metric
            for metric in collect_metric_keys(utility_rows)
            if metric not in {"utility_avg", "utility_delta_vs_base"}
        ]
        merged_by_label: dict[str, dict[str, Any]] = {}

        for row in checkpoint_rows:
            merged_by_label[row["label"]] = dict(row)

        for row in utility_rows:
            label = str(row["label"])
            merged = merged_by_label.setdefault(
                label,
                {
                    "label": label,
                    "checkpoint": label,
                    "step": None,
                    "epoch": None,
                    "forget_qa_rouge": None,
                    "holdout_qa_rouge": None,
                },
            )
            merged["checkpoint"] = coalesce(merged.get("checkpoint"), row.get("checkpoint"), label)
            merged["step"] = coalesce(merged.get("step"), row.get("step"))
            merged["epoch"] = coalesce(merged.get("epoch"), row.get("epoch"))
            for key in (*utility_metric_names, "utility_avg", "utility_delta_vs_base"):
                merged[key] = row.get(key)

        merged_rows = list(merged_by_label.values())

    wrong_generation_rows = collect_wrong_generation_rows(run_dir)
    if not wrong_generation_rows:
        return sorted(
            merged_rows,
            key=lambda row: checkpoint_sort_key(str(row["label"]), row.get("step")),
        )

    merged_by_label = {
        str(row["label"]): dict(row)
        for row in merged_rows
    }
    for row in wrong_generation_rows:
        label = str(row["label"])
        merged = merged_by_label.setdefault(
            label,
            {
                "label": label,
                "checkpoint": label,
                "step": row.get("step"),
                "epoch": row.get("epoch"),
            },
        )
        merged["checkpoint"] = coalesce(merged.get("checkpoint"), row.get("checkpoint"), label)
        merged["step"] = coalesce(merged.get("step"), row.get("step"))
        merged["epoch"] = coalesce(merged.get("epoch"), row.get("epoch"))
        merged["forget_wrong_gen_rate"] = row.get("forget_wrong_gen_rate")
        merged["holdout_wrong_gen_rate"] = row.get("holdout_wrong_gen_rate")

    return sorted(
        merged_by_label.values(),
        key=lambda row: checkpoint_sort_key(str(row["label"]), row.get("step")),
    )


def build_params_payload(
    run_dir: Path,
    split_bucket: str,
    lr: str,
    method_key: str,
    config: dict[str, Any],
    overrides: list[Any],
) -> dict[str, Any]:
    override_map = extract_override_map(overrides)
    trainer = config.get("trainer", {}) if isinstance(config.get("trainer"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    dataset_summary = {
        "forget_split": resolve_simple_reference(config.get("forget_split"), config),
        "retain_split": resolve_simple_reference(config.get("retain_split"), config),
        "holdout_split": resolve_simple_reference(config.get("holdout_split"), config),
        "question_key": resolve_simple_reference(config.get("question_key"), config),
    }

    return {
        "run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "split_bucket": split_bucket,
        "lr": lr,
        "method": method_key,
        "experiment": override_map.get("experiment"),
        "trainer_name": override_map.get("trainer"),
        "model_name": override_map.get("model"),
        "trainer_handler": trainer.get("handler"),
        "dataset": dataset_summary,
        "model_summary": {
            "use_lora": model.get("use_lora"),
            "pretrained_model_name_or_path": (
                model.get("model_args", {}).get("pretrained_model_name_or_path")
                if isinstance(model.get("model_args"), dict)
                else None
            ),
            "model_subfolder": (
                model.get("model_args", {}).get("subfolder")
                if isinstance(model.get("model_args"), dict)
                else None
            ),
            "tokenizer_pretrained_model_name_or_path": (
                model.get("tokenizer_args", {}).get("pretrained_model_name_or_path")
                if isinstance(model.get("tokenizer_args"), dict)
                else None
            ),
            "tokenizer_subfolder": (
                model.get("tokenizer_args", {}).get("subfolder")
                if isinstance(model.get("tokenizer_args"), dict)
                else None
            ),
            "lora_config": model.get("lora_config"),
        },
        "trainer_args": trainer.get("args"),
        "method_args": trainer.get("method_args"),
        "overrides": overrides,
        "hydra_config": config,
        "source_files": {
            "config_yaml": str(run_dir / ".hydra" / "config.yaml"),
            "overrides_yaml": (
                str(run_dir / ".hydra" / "overrides.yaml")
                if (run_dir / ".hydra" / "overrides.yaml").exists()
                else None
            ),
            "merged_summary_tsv": (
                str(run_dir / "checkpoint_evals_merged" / "summary.tsv")
                if (run_dir / "checkpoint_evals_merged" / "summary.tsv").exists()
                else None
            ),
            "trajectory_metrics_json": (
                str(run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json")
                if (run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json").exists()
                else None
            ),
        },
    }


def flatten_trajectory(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, subvalue in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_trajectory(next_prefix, subvalue))
        return flattened
    return {prefix: value}


def sanitize_metric_name(metric_name: str) -> str:
    return metric_name.replace("/", "_").replace(".", "_")


def run_seed_sort_key(run: dict[str, Any]) -> tuple[int, int, str]:
    _canonical_name, seed = split_seed_suffix(str(run["run_name"]))
    if seed is None:
        return (1, -1, str(run["run_name"]))
    return (0, int(seed), str(run["run_name"]))


def group_runs_for_output(
    runs: list[dict[str, Any]],
    *,
    average_seeds: bool,
) -> list[dict[str, Any]]:
    if not average_seeds:
        output_groups = []
        for run in sorted(runs, key=lambda item: method_sort_key(str(item["method"]))):
            _canonical_name, seed = split_seed_suffix(str(run["run_name"]))
            output_groups.append(
                {
                    "method": run["method"],
                    "run_name": run["run_name"],
                    "params_file": run["params_file"],
                    "runs": [run],
                    "seed_values": [] if seed is None else [seed],
                    "source_run_dirs": [str(run["run_dir"])],
                }
            )
        return output_groups

    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        method_name = str(run["method"])
        canonical_name, _seed = split_seed_suffix(str(run["run_name"]))
        group = grouped.setdefault(
            method_name,
            {
                "method": method_name,
                "run_name": canonical_name,
                "params_file": run["params_file"],
                "runs": [],
            },
        )
        if group["run_name"] != canonical_name:
            raise ValueError(
                f"Method {method_name} maps to multiple canonical run names: "
                f"{group['run_name']} vs {canonical_name}"
            )
        group["runs"].append(run)

    output_groups = sorted(grouped.values(), key=lambda item: method_sort_key(str(item["method"])))
    for group in output_groups:
        group["runs"].sort(key=run_seed_sort_key)
        group["seed_values"] = [
            seed
            for _canonical_name, seed in (split_seed_suffix(str(run["run_name"])) for run in group["runs"])
            if seed is not None
        ]
        group["source_run_dirs"] = [str(run["run_dir"]) for run in group["runs"]]
    return output_groups


def build_epoch_reference_rows(run_groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    slot_map: dict[str, dict[str, Any]] = {}
    for group in run_groups:
        for run in group["runs"]:
            for slot_row in build_epoch_slots(run["merged_rows"])[0]:
                slot_name = str(slot_row["slot"])
                current_row = slot_map.get(slot_name)
                if current_row is None or (
                    current_row.get("label") != "final" and slot_row.get("label") == "final"
                ):
                    slot_map[slot_name] = slot_row

    slot_names = sorted(slot_map, key=float)
    return [slot_map[slot_name] for slot_name in slot_names], slot_names


def collect_metric_names_for_runs(run_groups: list[dict[str, Any]], row_key: str) -> list[str]:
    metric_keys: set[str] = set()
    for group in run_groups:
        for run in group["runs"]:
            metric_keys.update(collect_metric_keys(run[row_key]))

    ordered = [metric for metric in PREFERRED_METRIC_ORDER if metric in metric_keys]
    utility_metric_keys = sorted(
        (metric for metric in metric_keys if UTILITY_COLUMN_RE.fullmatch(metric)),
        key=utility_metric_sort_key,
    )
    ordered.extend(metric for metric in utility_metric_keys if metric not in ordered)
    ordered.extend(sorted(metric_keys - set(ordered)))
    return ordered


def main() -> None:
    args = parse_args()
    unlearn_root = resolve_unlearn_root(args.input_root)
    output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)

    run_dirs = sorted(path for path in unlearn_root.iterdir() if path.is_dir())

    params_dir = output_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    grouped_runs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    params_index_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        config_path = run_dir / ".hydra" / "config.yaml"
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        trajectory_path = run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json"

        if not config_path.exists():
            continue

        config = load_yaml(config_path)
        overrides = (load_yaml(overrides_path) or []) if overrides_path.exists() else []
        merged_rows = collect_merged_rows(run_dir)
        if not merged_rows:
            continue
        cosine_rows = collect_cosine_rows(run_dir)
        trajectory_metrics = load_json(trajectory_path) if trajectory_path.exists() else {}

        split_bucket = infer_split_bucket(run_dir.name, config)
        lr = extract_lr(run_dir.name)
        method_key = extract_method_key(run_dir.name)
        params_payload = build_params_payload(run_dir, split_bucket, lr, method_key, config, overrides)

        params_file = params_dir / f"{run_dir.name}.yaml"
        with params_file.open("w", encoding="utf-8") as handle:
            yaml.dump(
                params_payload,
                handle,
                Dumper=NoAliasSafeDumper,
                allow_unicode=False,
                sort_keys=False,
                default_flow_style=False,
            )

        params_index_rows.append(
            {
                "split_bucket": split_bucket,
                "lr": lr,
                "method": method_key,
                "run_name": run_dir.name,
                "trainer_handler": params_payload.get("trainer_handler"),
                "experiment": params_payload.get("experiment"),
                "forget_split": params_payload["dataset"]["forget_split"],
                "retain_split": params_payload["dataset"]["retain_split"],
                "holdout_split": params_payload["dataset"]["holdout_split"],
                "params_file": str(params_file.relative_to(output_root)),
                "source_run_dir": str(run_dir),
            }
        )

        grouped_runs[(split_bucket, lr)].append(
            {
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "split_bucket": split_bucket,
                "lr": lr,
                "method": method_key,
                "params_file": params_file,
                "merged_rows": merged_rows,
                "cosine_rows": cosine_rows,
                "trajectory_metrics": trajectory_metrics,
            }
        )

    params_index_rows.sort(
        key=lambda row: (row["split_bucket"], lr_sort_key(row["lr"]), method_sort_key(row["method"]))
    )
    write_tsv(
        params_dir / "params_index.tsv",
        [
            "split_bucket",
            "lr",
            "method",
            "run_name",
            "trainer_handler",
            "experiment",
            "forget_split",
            "retain_split",
            "holdout_split",
            "params_file",
            "source_run_dir",
        ],
        params_index_rows,
    )

    for (split_bucket, lr), runs in sorted(
        grouped_runs.items(),
        key=lambda item: (item[0][0], lr_sort_key(item[0][1])),
    ):
        split_lr_root = output_root / split_bucket / lr
        split_lr_root.mkdir(parents=True, exist_ok=True)

        run_groups = group_runs_for_output(runs, average_seeds=args.average_seeds)
        epoch_rows, epoch_slots = build_epoch_reference_rows(run_groups)

        write_tsv(
            split_lr_root / "epoch_reference.tsv",
            ["slot", "label", "checkpoint", "step", "actual_epoch"],
            epoch_rows,
        )

        write_tsv(
            split_lr_root / "runs_index.tsv",
            ["method", "run_name", "params_file", "source_run_dir"],
            [
                {
                    "method": group["method"],
                    "run_name": group["run_name"],
                    "params_file": str(group["params_file"].relative_to(output_root)),
                    "source_run_dir": "|".join(group["source_run_dirs"]),
                }
                for group in run_groups
            ],
        )

        merged_metric_names = collect_metric_names_for_runs(run_groups, "merged_rows")
        cosine_metric_names = collect_metric_names_for_runs(run_groups, "cosine_rows")
        all_metric_names = merged_metric_names + [
            metric for metric in cosine_metric_names if metric not in merged_metric_names
        ]

        for metric_name in all_metric_names:
            fieldnames = ["method"] + epoch_slots
            table_rows: list[dict[str, Any]] = []
            for group in run_groups:
                metric_maps = [
                    build_metric_values_by_slot(
                        run["cosine_rows"] if metric_name.endswith("_cos_sim") else run["merged_rows"],
                        metric_name,
                    )
                    for run in group["runs"]
                ]
                row = {"method": group["method"]}
                for slot in epoch_slots:
                    row[slot] = average_optional_floats(
                        [metric_values_by_slot.get(slot) for metric_values_by_slot in metric_maps]
                    )
                table_rows.append(row)

            write_tsv(
                split_lr_root / f"{sanitize_metric_name(metric_name)}.tsv",
                fieldnames,
                table_rows,
            )

        trajectory_rows: list[dict[str, Any]] = []
        trajectory_fieldnames = ["method"]
        flattened_payloads: list[tuple[str, list[dict[str, Any]]]] = []
        for group in run_groups:
            flattened_runs = [flatten_trajectory("", run["trajectory_metrics"]) for run in group["runs"]]
            flattened_payloads.append((str(group["method"]), flattened_runs))
            for flattened in flattened_runs:
                for key in flattened:
                    if key not in trajectory_fieldnames:
                        trajectory_fieldnames.append(key)

        for method_name, flattened_runs in flattened_payloads:
            row = {"method": method_name}
            for key in trajectory_fieldnames[1:]:
                row[key] = average_scalar_values([flattened.get(key) for flattened in flattened_runs])
            trajectory_rows.append(row)

        write_tsv(
            split_lr_root / "trajectory_summary.tsv",
            trajectory_fieldnames,
            trajectory_rows,
        )


if __name__ == "__main__":
    main()
