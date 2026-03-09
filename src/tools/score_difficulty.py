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
        return {
            int(row["index"]): float(row[args.confidence_column])
            for row in dataset
            if args.confidence_column in row and row[args.confidence_column] is not None
        }

    if not args.model_cfg:
        return {}

    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
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
    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    dataloader = DataLoader(
        qa_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=collator,
    )

    confidence_scores: Dict[int, float] = {}
    with torch.inference_mode():
        for batch in dataloader:
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
    return confidence_scores


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
    stage_map = load_stage_map(args.stage_map_json)
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
    for idx, row in enumerate(rows):
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

    save_jsonl(output_rows, args.output_path)


if __name__ == "__main__":
    main()
