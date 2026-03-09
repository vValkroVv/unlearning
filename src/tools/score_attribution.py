#!/usr/bin/env python3
"""Compute proxy retain-gradient attribution scores for DualCF."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader

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


def parse_args():
    parser = argparse.ArgumentParser(description="Score DualCF attribution artifacts.")
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--forget-dataset-path", required=True)
    parser.add_argument("--forget-split", required=True)
    parser.add_argument("--forget-dataset-name", default=None)
    parser.add_argument("--forget-data-files", default=None)
    parser.add_argument("--retain-dataset-path", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--retain-dataset-name", default=None)
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--forget-answer-key", default="answer")
    parser.add_argument("--forget-answer-index", type=int, default=None)
    parser.add_argument("--retain-question-key", default=None)
    parser.add_argument("--retain-answer-key", default="answer")
    parser.add_argument("--retain-answer-index", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retain-batch-size", type=int, default=1)
    parser.add_argument("--retain-max-steps", type=int, default=64)
    parser.add_argument("--forget-max-examples", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora-only", action="store_true")
    parser.add_argument("--alignment", choices=("dot", "cosine"), default="dot")
    return parser.parse_args()


def iter_selected_params(model, lora_only: bool) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if lora_only and "lora_" not in name.lower():
            continue
        yield name, param


def to_model_inputs(batch, device: str, move_to_device: bool):
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    if move_to_device:
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    return model_inputs


def accumulate_retain_gradient(
    model,
    dataloader,
    selected_params,
    device: str,
    move_to_device: bool,
    max_steps: int,
):
    retain_grad = {
        name: torch.zeros_like(param.detach().float(), device="cpu")
        for name, param in selected_params
    }
    steps = 0
    for batch in dataloader:
        if max_steps > 0 and steps >= max_steps:
            break
        model.zero_grad(set_to_none=True)
        outputs = model(**to_model_inputs(batch, device=device, move_to_device=move_to_device))
        outputs.loss.backward()
        for name, param in selected_params:
            if param.grad is not None:
                retain_grad[name] += param.grad.detach().float().cpu()
        steps += 1

    if steps <= 0:
        raise ValueError("Retain gradient accumulation saw zero steps.")
    for name in retain_grad:
        retain_grad[name] /= float(steps)
    retain_norm = math.sqrt(
        sum(float((grad * grad).sum().item()) for grad in retain_grad.values())
    )
    return retain_grad, retain_norm


def main():
    args = parse_args()
    retain_question_key = args.retain_question_key or args.question_key

    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
    )
    device = select_device(args.device)
    move_to_device = getattr(model, "hf_device_map", None) is None
    if move_to_device:
        model = model.to(device)
    model.eval()

    forget_dataset = build_qa_dataset(
        dataset_path=args.forget_dataset_path,
        split=args.forget_split,
        tokenizer=tokenizer,
        template_args=template_args,
        question_key=args.question_key,
        answer_key=args.forget_answer_key,
        answer_index=args.forget_answer_index,
        max_length=args.max_length,
        name=args.forget_dataset_name,
        data_files=args.forget_data_files,
    )
    retain_dataset = build_qa_dataset(
        dataset_path=args.retain_dataset_path,
        split=args.retain_split,
        tokenizer=tokenizer,
        template_args=template_args,
        question_key=retain_question_key,
        answer_key=args.retain_answer_key,
        answer_index=args.retain_answer_index,
        max_length=args.max_length,
        name=args.retain_dataset_name,
        data_files=args.retain_data_files,
    )
    forget_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.forget_dataset_path,
            split=args.forget_split,
            name=args.forget_dataset_name,
            data_files=args.forget_data_files,
            max_examples=args.forget_max_examples,
        )
    ]
    if args.forget_max_examples and args.forget_max_examples > 0:
        forget_dataset = torch.utils.data.Subset(
            forget_dataset,
            range(min(int(args.forget_max_examples), len(forget_dataset))),
        )

    collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=max(1, int(args.retain_batch_size)),
        shuffle=False,
        collate_fn=collator,
    )
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    selected_params = list(iter_selected_params(model, lora_only=args.lora_only))
    if not selected_params:
        raise ValueError("No trainable parameters matched the requested attribution setup.")

    retain_grad, retain_norm = accumulate_retain_gradient(
        model=model,
        dataloader=retain_loader,
        selected_params=selected_params,
        device=device,
        move_to_device=move_to_device,
        max_steps=int(args.retain_max_steps),
    )

    grad_by_index: Dict[int, Dict[str, float]] = {}
    for batch in forget_loader:
        model.zero_grad(set_to_none=True)
        outputs = model(**to_model_inputs(batch, device=device, move_to_device=move_to_device))
        outputs.loss.backward()

        dot_value = 0.0
        forget_norm_sq = 0.0
        for name, param in selected_params:
            if param.grad is None:
                continue
            forget_grad = param.grad.detach().float().cpu()
            retain_component = retain_grad[name]
            dot_value += float((forget_grad * retain_component).sum().item())
            forget_norm_sq += float((forget_grad * forget_grad).sum().item())

        forget_norm = math.sqrt(forget_norm_sq)
        cosine_value = dot_value / max(forget_norm * max(retain_norm, 1e-12), 1e-12)
        score_value = cosine_value if args.alignment == "cosine" else dot_value
        grad_by_index[int(batch["index"][0].item())] = {
            "grad_align": dot_value,
            "grad_align_cosine": cosine_value,
            "risk_raw": max(0.0, score_value),
        }

    risk_norm = normalize_minmax(
        [grad_by_index[int(row["index"])]["risk_raw"] for row in forget_rows]
    )

    output_rows = []
    for norm_score, row in zip(risk_norm, forget_rows):
        row_index = int(row["index"])
        answer = resolve_answer(
            row=row,
            answer_key=args.forget_answer_key,
            answer_index=args.forget_answer_index,
        )
        updated = dict(row)
        updated["answer"] = answer
        updated["grad_align"] = grad_by_index[row_index]["grad_align"]
        updated["grad_align_cosine"] = grad_by_index[row_index]["grad_align_cosine"]
        updated["attribution_score"] = norm_score
        output_rows.append(updated)

    save_jsonl(output_rows, args.output_path)


if __name__ == "__main__":
    main()
