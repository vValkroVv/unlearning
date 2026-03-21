#!/usr/bin/env python3
"""Build comparison tables from packaged unlearning save summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from checkpoint_summary_utils import collect_eval_summaries


META_COLUMNS = {"label", "checkpoint", "step", "epoch"}
PREFERRED_METRIC_ORDER = [
    "forget_qa_rouge",
    "holdout_qa_rouge",
    "forget_qa_cos_sim",
    "holdout_qa_cos_sim",
    "utility_avg",
    "utility_delta_vs_base",
    "mmlu_pro_400_acc",
    "truthfulqa_bin_200_acc",
    "arc_200_acc",
    "winogrande_200_acc",
]
METHOD_ORDER = [
    "full",
    "d_only",
    "a_only",
    "dpo",
    "simple_ce",
    "ga",
    "npo",
    "simnpo",
    "npo_sam",
    "loku",
]
METHOD_ORDER_INDEX = {name: index for index, name in enumerate(METHOD_ORDER)}
LR_RE = re.compile(r"_lr([^_]+)")
METHOD_RE = re.compile(r"_(dual_cf|dpo_cf|simple_ce|simnpo|ga|loku|npo_sam|npo)_lora_.*?_lr[^_]+(.*)$")
DUAL_FLAG_RE = re.compile(r"^(dOn|dOff|aOn|aOff|adT|adF)$")


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
    if method_name == "npo_sam":
        return "npo_sam"
    if method_name == "simple_ce":
        return "simple_ce"
    if method_name == "simnpo":
        return "simnpo"
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


def build_epoch_slots(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    slot_rows: list[dict[str, Any]] = []
    slot_names: list[str] = []

    representatives = dedupe_representative_rows(rows)
    slot_index = 0
    for row in representatives:
        label = row.get("label")
        if label == "base_model_orig":
            slot_name = "0.0"
        else:
            slot_index += 1
            slot_name = format(slot_index * 0.5, ".1f")
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
    return slot_rows, slot_names


def collect_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    metric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if key not in META_COLUMNS and value is not None
    }
    ordered = [metric for metric in PREFERRED_METRIC_ORDER if metric in metric_keys]
    ordered.extend(sorted(metric_keys - set(ordered)))
    return ordered


def method_sort_key(method_name: str) -> tuple[int, str]:
    return (METHOD_ORDER_INDEX.get(method_name, len(METHOD_ORDER)), method_name)


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
            "overrides_yaml": str(run_dir / ".hydra" / "overrides.yaml"),
            "merged_summary_tsv": str(run_dir / "checkpoint_evals_merged" / "summary.tsv"),
            "trajectory_metrics_json": str(run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json"),
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
        merged_summary_path = run_dir / "checkpoint_evals_merged" / "summary.tsv"
        trajectory_path = run_dir / "checkpoint_evals_merged" / "trajectory_metrics.json"

        if not config_path.exists() or not overrides_path.exists() or not merged_summary_path.exists():
            continue

        config = load_yaml(config_path)
        overrides = load_yaml(overrides_path) or []
        merged_rows = load_tsv_rows(merged_summary_path)
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

        runs.sort(key=lambda row: method_sort_key(row["method"]))
        epoch_rows, epoch_slots = build_epoch_slots(runs[0]["merged_rows"])

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
                    "method": run["method"],
                    "run_name": run["run_name"],
                    "params_file": str(run["params_file"].relative_to(output_root)),
                    "source_run_dir": str(run["run_dir"]),
                }
                for run in runs
            ],
        )

        merged_metric_names = collect_metric_keys(runs[0]["merged_rows"])
        cosine_metric_names = collect_metric_keys(runs[0]["cosine_rows"])
        all_metric_names = merged_metric_names + [
            metric for metric in cosine_metric_names if metric not in merged_metric_names
        ]

        for metric_name in all_metric_names:
            fieldnames = ["method"] + epoch_slots
            table_rows: list[dict[str, Any]] = []
            for run in runs:
                source_rows = run["cosine_rows"] if metric_name.endswith("_cos_sim") else run["merged_rows"]
                slot_rows, _ = build_epoch_slots(source_rows)
                metric_values_by_slot = {
                    slot_row["slot"]: source_row.get(metric_name)
                    for slot_row, source_row in zip(slot_rows, dedupe_representative_rows(source_rows))
                }
                row = {"method": run["method"]}
                for slot in epoch_slots:
                    row[slot] = metric_values_by_slot.get(slot)
                table_rows.append(row)

            write_tsv(
                split_lr_root / f"{sanitize_metric_name(metric_name)}.tsv",
                fieldnames,
                table_rows,
            )

        trajectory_rows: list[dict[str, Any]] = []
        trajectory_fieldnames = ["method"]
        flattened_payloads: list[tuple[str, dict[str, Any]]] = []
        for run in runs:
            flattened = flatten_trajectory("", run["trajectory_metrics"])
            flattened_payloads.append((run["method"], flattened))
            for key in flattened:
                if key not in trajectory_fieldnames:
                    trajectory_fieldnames.append(key)

        for method_name, flattened in flattened_payloads:
            row = {"method": method_name}
            row.update(flattened)
            trajectory_rows.append(row)

        write_tsv(
            split_lr_root / "trajectory_summary.tsv",
            trajectory_fieldnames,
            trajectory_rows,
        )


if __name__ == "__main__":
    main()
