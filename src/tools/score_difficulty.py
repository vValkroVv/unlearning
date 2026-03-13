#!/usr/bin/env python3
"""Score DualCF difficulty proxies and emit a merged artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.collators import DataCollatorForSupervisedDataset
from tools.dual_cf_artifact_utils import (
    build_qa_dataset,
    load_dataset_split,
    load_model_bundle,
    normalize_minmax,
    resolve_answer,
    save_jsonl,
    select_device,
)
from trainer.utils import compute_nll_per_sample


def log(message: str) -> None:
    print(f"[score_difficulty] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Score DualCF difficulty artifacts.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--answer-index", type=int, default=None)
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--tokenizer-subfolder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--mrd-column", default=None)
    parser.add_argument("--popularity-column", default="pop_sum")
    parser.add_argument("--confidence-column", default=None)
    parser.add_argument("--stage-column", default=None)
    parser.add_argument("--stage-map-json", default=None)
    parser.add_argument("--stage-prior-constant", type=float, default=None)
    parser.add_argument("--w-mrd", type=float, default=0.0)
    parser.add_argument("--w-pop", type=float, default=1.0)
    parser.add_argument("--w-conf", type=float, default=1.0)
    parser.add_argument("--w-stage", type=float, default=0.0)
    return parser.parse_args()


def load_stage_map(raw: Optional[str]) -> Optional[Dict[str, float]]:
    if raw in (None, "", "null", "None"):
        return None
    with open(raw, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(k): float(v) for k, v in payload.items()}


def collect_confidence_scores(args, dataset) -> Dict[int, float]:
    if args.confidence_column:
        log(f"Using precomputed confidence column `{args.confidence_column}`")
        return {
            int(row["index"]): float(row[args.confidence_column])
            for row in dataset
            if args.confidence_column in row and row[args.confidence_column] is not None
        }

    if not args.model_cfg:
        log("No model config provided, skipping confidence scoring")
        return {}

    log(
        "Loading confidence model "
        f"cfg={args.model_cfg} model_path={args.model_path or '<cfg-default>'} "
        f"tokenizer_path={args.tokenizer_path or '<model-path>'}"
    )
    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_subfolder=args.model_subfolder,
        tokenizer_subfolder=args.tokenizer_subfolder,
    )
    device = select_device(args.device)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()

    qa_dataset = build_qa_dataset(
        dataset_path=args.dataset_path,
        split=args.split,
        tokenizer=tokenizer,
        template_args=template_args,
        question_key=args.question_key,
        answer_key=args.answer_key,
        answer_index=args.answer_index,
        max_length=args.max_length,
        name=args.dataset_name,
        data_files=args.data_files,
    )
    if args.max_examples and args.max_examples > 0:
        qa_dataset = Subset(
            qa_dataset, range(min(int(args.max_examples), len(qa_dataset)))
        )
    qa_size = len(qa_dataset) if hasattr(qa_dataset, "__len__") else None
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    dataloader = DataLoader(
        qa_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=collator,
    )
    log(
        "Scoring confidence with "
        f"rows={qa_size if qa_size is not None else 'unknown'} "
        f"batch_size={max(1, int(args.batch_size))} device={device}"
    )

    confidence_scores: Dict[int, float] = {}
    with torch.inference_mode():
        batch_iter = tqdm(
            dataloader,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc="confidence",
            dynamic_ncols=True,
        )
        for batch in batch_iter:
            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            if getattr(model, "hf_device_map", None) is None:
                model_inputs = {
                    key: value.to(device) for key, value in model_inputs.items()
                }
            losses, _ = compute_nll_per_sample(
                model, model_inputs, normalize_by_tokens=True
            )
            for sample_index, loss in zip(batch["index"].tolist(), losses.tolist()):
                confidence_scores[int(sample_index)] = float(-loss)
    log(f"Collected confidence scores for {len(confidence_scores)} rows")
    return confidence_scores


def main():
    args = parse_args()
    log(
        "Starting with "
        f"dataset_path={args.dataset_path} split={args.split} "
        f"dataset_name={args.dataset_name} data_files={args.data_files} "
        f"output_path={args.output_path}"
    )
    dataset = load_dataset_split(
        path=args.dataset_path,
        split=args.split,
        name=args.dataset_name,
        data_files=args.data_files,
        max_examples=args.max_examples,
    )
    rows = [dict(row) for row in dataset]
    log(f"Loaded {len(rows)} rows for difficulty scoring")
    stage_map = load_stage_map(args.stage_map_json)
    if stage_map is not None:
        log(f"Loaded stage map with {len(stage_map)} entries from {args.stage_map_json}")
    confidence_scores = collect_confidence_scores(args=args, dataset=dataset)

    mrd_raw = []
    pop_raw = []
    conf_raw = []
    stage_raw = []
    active_weights = []

    has_mrd = bool(args.mrd_column) and any(args.mrd_column in row for row in rows)
    has_pop = bool(args.popularity_column) and any(
        args.popularity_column in row for row in rows
    )
    has_stage = bool(args.stage_column) and any(args.stage_column in row for row in rows)

    if has_mrd:
        mrd_raw = [
            float(row[args.mrd_column]) if row.get(args.mrd_column) is not None else 0.0
            for row in rows
        ]
        active_weights.append(float(args.w_mrd))
    if has_pop:
        pop_raw = [
            float(row[args.popularity_column])
            if row.get(args.popularity_column) is not None
            else 0.0
            for row in rows
        ]
        active_weights.append(float(args.w_pop))
    if confidence_scores:
        conf_raw = [float(confidence_scores[int(row["index"])]) for row in rows]
        active_weights.append(float(args.w_conf))
    if has_stage or args.stage_prior_constant is not None:
        if has_stage and stage_map is not None:
            stage_raw = [
                float(stage_map.get(str(row.get(args.stage_column)), 0.0)) for row in rows
            ]
        else:
            stage_value = float(args.stage_prior_constant or 0.0)
            stage_raw = [stage_value for _ in rows]
        active_weights.append(float(args.w_stage))

    if not any(weight > 0.0 for weight in active_weights):
        raise ValueError(
            "No active difficulty components. Provide at least one positive weight and source."
        )

    active_components = []
    if has_mrd and args.w_mrd > 0.0:
        active_components.append(f"mrd(w={args.w_mrd})")
    if has_pop and args.w_pop > 0.0:
        active_components.append(f"popularity(w={args.w_pop})")
    if confidence_scores and args.w_conf > 0.0:
        active_components.append(f"confidence(w={args.w_conf})")
    if stage_raw and args.w_stage > 0.0:
        active_components.append(f"stage(w={args.w_stage})")
    log("Active difficulty components: " + ", ".join(active_components))

    hardness_mrd = []
    pop_norm = []
    conf_norm = []
    if mrd_raw:
        hardness_mrd = [1.0 - value for value in normalize_minmax(mrd_raw)]
    if pop_raw:
        pop_norm = normalize_minmax(pop_raw)
    if conf_raw:
        conf_norm = normalize_minmax(conf_raw)
    stage_norm = stage_raw

    output_rows = []
    row_iter = tqdm(
        enumerate(rows),
        total=len(rows),
        desc="difficulty_rows",
        dynamic_ncols=True,
    )
    for idx, row in row_iter:
        row_index = row.get("index", "<missing>")
        try:
            weighted_sum = 0.0
            weight_total = 0.0

            row_answer = resolve_answer(
                row=row,
                answer_key=args.answer_key,
                answer_index=args.answer_index,
            )
            updated = dict(row)
            updated["answer"] = row_answer

            if hardness_mrd and args.w_mrd > 0.0:
                updated["hardness_mrd"] = hardness_mrd[idx]
                weighted_sum += float(args.w_mrd) * hardness_mrd[idx]
                weight_total += float(args.w_mrd)
            if pop_norm and args.w_pop > 0.0:
                updated["difficulty_popularity_norm"] = pop_norm[idx]
                weighted_sum += float(args.w_pop) * pop_norm[idx]
                weight_total += float(args.w_pop)
            if conf_norm and args.w_conf > 0.0:
                updated["difficulty_confidence_norm"] = conf_norm[idx]
                weighted_sum += float(args.w_conf) * conf_norm[idx]
                weight_total += float(args.w_conf)
            if stage_norm and args.w_stage > 0.0:
                updated["difficulty_stage_prior"] = float(stage_norm[idx])
                weighted_sum += float(args.w_stage) * float(stage_norm[idx])
                weight_total += float(args.w_stage)

            if weight_total <= 0.0:
                raise ValueError("Difficulty score has zero active weight for at least one row.")
            updated["difficulty_score"] = weighted_sum / weight_total
            output_rows.append(updated)
        except Exception as exc:
            raise RuntimeError(
                f"Failed processing difficulty row index={row_index} at position={idx}"
            ) from exc

    difficulty_scores = [row["difficulty_score"] for row in output_rows]
    log(f"Saving {len(output_rows)} rows to {args.output_path}")
    save_jsonl(output_rows, args.output_path)
    log(
        "Done. "
        f"difficulty_score_range=({min(difficulty_scores):.6f}, {max(difficulty_scores):.6f})"
    )


if __name__ == "__main__":
    main()
