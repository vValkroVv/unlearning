#!/usr/bin/env python3
"""Compute proxy retain-gradient attribution scores for DualCF."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    load_keyed_jsonish,
    load_model_bundle,
    normalize_minmax,
    resolve_answer,
    save_jsonl,
    select_device,
)


def log(message: str) -> None:
    print(f"[score_attribution] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Score DualCF attribution artifacts.")
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--tokenizer-subfolder", default=None)
    parser.add_argument("--forget-dataset-path", required=True)
    parser.add_argument("--forget-split", required=True)
    parser.add_argument("--forget-dataset-name", default=None)
    parser.add_argument("--forget-data-files", default=None)
    parser.add_argument("--retain-dataset-path", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--retain-dataset-name", default=None)
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--utility-dataset-path", default=None)
    parser.add_argument("--utility-split", default="train")
    parser.add_argument("--utility-dataset-name", default=None)
    parser.add_argument("--utility-data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--forget-answer-key", default="answer")
    parser.add_argument("--forget-answer-index", type=int, default=None)
    parser.add_argument("--retain-question-key", default=None)
    parser.add_argument("--retain-answer-key", default="answer")
    parser.add_argument("--retain-answer-index", type=int, default=None)
    parser.add_argument("--utility-question-key", default="question")
    parser.add_argument("--utility-answer-key", default="answer")
    parser.add_argument("--utility-answer-index", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--retain-batch-size", type=int, default=1)
    parser.add_argument("--retain-max-steps", type=int, default=64)
    parser.add_argument("--forget-max-examples", type=int, default=0)
    parser.add_argument("--forget-max-steps", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora-only", action="store_true")
    parser.add_argument("--alignment", choices=("dot", "cosine"), default="dot")
    parser.add_argument(
        "--retain-proxy-mode",
        choices=("global", "template_local", "hybrid", "multi_bank"),
        default="global",
    )
    parser.add_argument("--retain-proxy-map", default=None)
    parser.add_argument("--hybrid-rho", type=float, default=0.7)
    parser.add_argument("--proxy-weight-global", type=float, default=0.15)
    parser.add_argument("--proxy-weight-syntax", type=float, default=0.45)
    parser.add_argument("--proxy-weight-semantic", type=float, default=0.25)
    parser.add_argument("--proxy-weight-utility", type=float, default=0.15)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
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
    total_steps = None
    if hasattr(dataloader, "__len__"):
        total_steps = len(dataloader)
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
    retain_iter = tqdm(
        dataloader,
        total=total_steps,
        desc="retain_grad",
        dynamic_ncols=True,
    )
    for batch in retain_iter:
        if max_steps > 0 and steps >= max_steps:
            break
        model.zero_grad(set_to_none=True)
        outputs = model(**to_model_inputs(batch, device=device, move_to_device=move_to_device))
        outputs.loss.backward()
        for name, param in selected_params:
            if param.grad is not None:
                retain_grad[name] += param.grad.detach().float().cpu()
        steps += 1
        retain_iter.set_postfix(steps=steps)

    if steps <= 0:
        raise ValueError("Retain gradient accumulation saw zero steps.")
    for name in retain_grad:
        retain_grad[name] /= float(steps)
    retain_norm = math.sqrt(
        sum(float((grad * grad).sum().item()) for grad in retain_grad.values())
    )
    return retain_grad, retain_norm, steps


def measure_alignment(
    forget_grads: Dict[str, torch.Tensor],
    retain_grad: Dict[str, torch.Tensor],
    retain_norm: float,
) -> Tuple[float, float]:
    dot_value = 0.0
    forget_norm_sq = 0.0
    for name, forget_grad in forget_grads.items():
        retain_component = retain_grad[name]
        dot_value += float((forget_grad * retain_component).sum().item())
        forget_norm_sq += float((forget_grad * forget_grad).sum().item())
    forget_norm = math.sqrt(forget_norm_sq)
    cosine_value = dot_value / max(forget_norm * max(retain_norm, 1e-12), 1e-12)
    return dot_value, cosine_value


def build_subset_loader(dataset, positions, batch_size, collator):
    subset = Subset(dataset, positions)
    return DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collator,
    )


def cached_alignment(
    *,
    cache: dict,
    cache_key: str,
    dataset,
    positions: list[int],
    batch_size: int,
    collator,
    model,
    selected_params,
    device: str,
    move_to_device: bool,
    max_steps: int,
    forget_grads: dict,
    alignment: str,
):
    if not positions:
        return 0.0, 0.0, 0.0
    if cache_key not in cache:
        loader = build_subset_loader(
            dataset=dataset,
            positions=positions,
            batch_size=batch_size,
            collator=collator,
        )
        cache[cache_key] = accumulate_retain_gradient(
            model=model,
            dataloader=loader,
            selected_params=selected_params,
            device=device,
            move_to_device=move_to_device,
            max_steps=max_steps,
        )
    retain_grad, retain_norm, _ = cache[cache_key]
    dot_value, cosine_value = measure_alignment(
        forget_grads=forget_grads,
        retain_grad=retain_grad,
        retain_norm=retain_norm,
    )
    raw_value = max(0.0, cosine_value if alignment == "cosine" else dot_value)
    return dot_value, cosine_value, raw_value


def main():
    args = parse_args()
    log(
        "Starting with "
        f"forget_dataset={args.forget_dataset_path}:{args.forget_dataset_name}:{args.forget_split} "
        f"retain_dataset={args.retain_dataset_path}:{args.retain_dataset_name}:{args.retain_split} "
        f"utility_dataset={args.utility_dataset_path}:{args.utility_dataset_name}:{args.utility_split} "
        f"output_path={args.output_path}"
    )
    retain_question_key = args.retain_question_key or args.question_key
    forget_limit = 0
    if int(args.forget_max_steps) > 0:
        forget_limit = int(args.forget_max_steps)
    elif int(args.forget_max_examples) > 0:
        forget_limit = int(args.forget_max_examples)

    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_subfolder=args.model_subfolder,
        tokenizer_subfolder=args.tokenizer_subfolder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    device = select_device(args.device)
    move_to_device = getattr(model, "hf_device_map", None) is None
    if move_to_device:
        model = model.to(device)
    model.eval()
    log(f"Model ready on device={device}")

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
    utility_dataset = None
    utility_rows: list[dict] = []
    if args.utility_dataset_path not in (None, "", "null", "None"):
        utility_dataset = build_qa_dataset(
            dataset_path=args.utility_dataset_path,
            split=args.utility_split,
            tokenizer=tokenizer,
            template_args=template_args,
            question_key=args.utility_question_key,
            answer_key=args.utility_answer_key,
            answer_index=args.utility_answer_index,
            max_length=args.max_length,
            name=args.utility_dataset_name,
            data_files=args.utility_data_files,
        )
        utility_rows = [
            dict(row)
            for row in load_dataset_split(
                path=args.utility_dataset_path,
                split=args.utility_split,
                name=args.utility_dataset_name,
                data_files=args.utility_data_files,
            )
        ]

    forget_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.forget_dataset_path,
            split=args.forget_split,
            name=args.forget_dataset_name,
            data_files=args.forget_data_files,
            max_examples=forget_limit,
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
    if forget_limit > 0:
        forget_dataset = torch.utils.data.Subset(
            forget_dataset,
            range(min(forget_limit, len(forget_dataset))),
        )
    log(
        "Prepared datasets with "
        f"forget_rows={len(forget_rows)} retain_rows={len(retain_rows)} "
        f"utility_rows={len(utility_rows)}"
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
    utility_collator = collator

    retain_position_by_index = {int(row["index"]): pos for pos, row in enumerate(retain_rows)}
    utility_position_by_index = {
        int(row["index"]): pos for pos, row in enumerate(utility_rows)
    }
    proxy_map = {}
    if args.retain_proxy_mode != "global":
        if args.retain_proxy_map in (None, "", "null", "None"):
            raise ValueError(
                "--retain-proxy-map is required when --retain-proxy-mode is not global"
            )
        proxy_map = load_keyed_jsonish(args.retain_proxy_map, key_field="index")
        log(f"Loaded retain proxy map rows={len(proxy_map)} mode={args.retain_proxy_mode}")

    selected_params = list(iter_selected_params(model, lora_only=args.lora_only))
    if not selected_params:
        raise ValueError("No trainable parameters matched the requested attribution setup.")

    retain_grad, retain_norm, retain_steps = accumulate_retain_gradient(
        model=model,
        dataloader=retain_loader,
        selected_params=selected_params,
        device=device,
        move_to_device=move_to_device,
        max_steps=int(args.retain_max_steps),
    )
    log(
        "Retain gradient accumulation finished "
        f"steps={retain_steps} retain_norm={retain_norm:.6f}"
    )

    grad_by_index: Dict[int, Dict[str, float]] = {}
    local_grad_cache: Dict[str, Tuple[Dict[str, torch.Tensor], float, int]] = {}
    forget_iter = tqdm(
        forget_loader,
        total=len(forget_loader),
        desc="forget_grad",
        dynamic_ncols=True,
    )
    for batch in forget_iter:
        row_index = int(batch["index"][0].item())
        try:
            model.zero_grad(set_to_none=True)
            outputs = model(**to_model_inputs(batch, device=device, move_to_device=move_to_device))
            outputs.loss.backward()

            forget_grads = {}
            for name, param in selected_params:
                if param.grad is None:
                    continue
                forget_grads[name] = param.grad.detach().float().cpu()

            global_dot, global_cosine = measure_alignment(
                forget_grads=forget_grads,
                retain_grad=retain_grad,
                retain_norm=retain_norm,
            )
            global_raw = max(0.0, global_cosine if args.alignment == "cosine" else global_dot)

            proxy_row = proxy_map.get(str(row_index), {})
            syntax_indices = [
                idx
                for idx in proxy_row.get("syntax_retain_indices", proxy_row.get("retain_indices", []))
                if int(idx) in retain_position_by_index
            ]
            semantic_indices = [
                idx
                for idx in proxy_row.get("semantic_retain_indices", [])
                if int(idx) in retain_position_by_index
            ]
            utility_indices = [
                idx
                for idx in proxy_row.get("utility_anchor_indices", [])
                if int(idx) in utility_position_by_index
            ]

            syntax_dot, syntax_cos, syntax_raw = cached_alignment(
                cache=local_grad_cache,
                cache_key=f"syn::{row_index}",
                dataset=retain_dataset,
                positions=[retain_position_by_index[int(idx)] for idx in syntax_indices],
                batch_size=args.retain_batch_size,
                collator=collator,
                model=model,
                selected_params=selected_params,
                device=device,
                move_to_device=move_to_device,
                max_steps=int(args.retain_max_steps),
                forget_grads=forget_grads,
                alignment=args.alignment,
            )
            semantic_dot, semantic_cos, semantic_raw = cached_alignment(
                cache=local_grad_cache,
                cache_key=f"sem::{row_index}",
                dataset=retain_dataset,
                positions=[retain_position_by_index[int(idx)] for idx in semantic_indices],
                batch_size=args.retain_batch_size,
                collator=collator,
                model=model,
                selected_params=selected_params,
                device=device,
                move_to_device=move_to_device,
                max_steps=int(args.retain_max_steps),
                forget_grads=forget_grads,
                alignment=args.alignment,
            )
            utility_dot, utility_cos, utility_raw = 0.0, 0.0, 0.0
            if utility_dataset is not None:
                utility_dot, utility_cos, utility_raw = cached_alignment(
                    cache=local_grad_cache,
                    cache_key=f"util::{row_index}",
                    dataset=utility_dataset,
                    positions=[utility_position_by_index[int(idx)] for idx in utility_indices],
                    batch_size=args.retain_batch_size,
                    collator=utility_collator,
                    model=model,
                    selected_params=selected_params,
                    device=device,
                    move_to_device=move_to_device,
                    max_steps=int(args.retain_max_steps),
                    forget_grads=forget_grads,
                    alignment=args.alignment,
                )

            has_multi_bank = bool(
                syntax_indices
                or semantic_indices
                or utility_indices
                or utility_dataset is not None
                or "syntax_retain_indices" in proxy_row
                or "semantic_retain_indices" in proxy_row
                or "utility_anchor_indices" in proxy_row
            )
            if args.retain_proxy_mode == "global":
                score_value = global_raw
                chosen_dot = global_dot
                chosen_cosine = global_cosine
                proxy_mode = "global"
                proxy_key = None
            elif args.retain_proxy_mode == "template_local":
                score_value = syntax_raw
                chosen_dot = syntax_dot
                chosen_cosine = syntax_cos
                proxy_mode = str(proxy_row.get("proxy_mode", "template_local"))
                proxy_key = str(proxy_row.get("template_key") or row_index)
            elif args.retain_proxy_mode == "hybrid" and not has_multi_bank:
                score_value = float(args.hybrid_rho) * syntax_raw + (
                    1.0 - float(args.hybrid_rho)
                ) * global_raw
                chosen_dot = float(args.hybrid_rho) * syntax_dot + (
                    1.0 - float(args.hybrid_rho)
                ) * global_dot
                chosen_cosine = float(args.hybrid_rho) * syntax_cos + (
                    1.0 - float(args.hybrid_rho)
                ) * global_cosine
                proxy_mode = str(proxy_row.get("proxy_mode", "hybrid"))
                proxy_key = str(proxy_row.get("template_key") or row_index)
            else:
                parts = [
                    ("global", global_raw, global_dot, global_cosine, float(args.proxy_weight_global)),
                    ("syntax", syntax_raw, syntax_dot, syntax_cos, float(args.proxy_weight_syntax)),
                    (
                        "semantic",
                        semantic_raw,
                        semantic_dot,
                        semantic_cos,
                        float(args.proxy_weight_semantic),
                    ),
                ]
                if utility_dataset is not None:
                    parts.append(
                        (
                            "utility",
                            utility_raw,
                            utility_dot,
                            utility_cos,
                            float(args.proxy_weight_utility),
                        )
                    )
                parts = [
                    (name, raw_value, dot_value, cosine_value, weight)
                    for name, raw_value, dot_value, cosine_value, weight in parts
                    if weight > 0.0
                ]
                denom = sum(weight for _, _, _, _, weight in parts) or 1.0
                score_value = sum(raw_value * weight for _, raw_value, _, _, weight in parts) / denom
                chosen_dot = sum(dot_value * weight for _, _, dot_value, _, weight in parts) / denom
                chosen_cosine = (
                    sum(cosine_value * weight for _, _, _, cosine_value, weight in parts) / denom
                )
                proxy_mode = "multi_bank_v3"
                proxy_key = str(proxy_row.get("template_key") or row_index)

            grad_by_index[row_index] = {
                "grad_align": chosen_dot,
                "grad_align_cosine": chosen_cosine,
                "global_grad_align": global_dot,
                "global_grad_align_cosine": global_cosine,
                "global_risk_raw": global_raw,
                "syntax_grad_align": syntax_dot,
                "syntax_grad_align_cosine": syntax_cos,
                "syntax_risk_raw": syntax_raw,
                "semantic_grad_align": semantic_dot,
                "semantic_grad_align_cosine": semantic_cos,
                "semantic_risk_raw": semantic_raw,
                "utility_grad_align": utility_dot,
                "utility_grad_align_cosine": utility_cos,
                "utility_risk_raw": utility_raw,
                "risk_raw": max(0.0, score_value),
                "proxy_mode": proxy_mode,
                "proxy_key": proxy_key,
                "syntax_size": len(syntax_indices),
                "semantic_size": len(semantic_indices),
                "utility_size": len(utility_indices),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed attribution scoring for forget index={row_index}") from exc

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
        updated["attribution_components"] = {
            "global_align": grad_by_index[row_index]["global_grad_align"],
            "global_align_cosine": grad_by_index[row_index]["global_grad_align_cosine"],
            "global_risk_raw": grad_by_index[row_index]["global_risk_raw"],
            "syntax_align": grad_by_index[row_index]["syntax_grad_align"],
            "syntax_align_cosine": grad_by_index[row_index]["syntax_grad_align_cosine"],
            "syntax_risk_raw": grad_by_index[row_index]["syntax_risk_raw"],
            "semantic_align": grad_by_index[row_index]["semantic_grad_align"],
            "semantic_align_cosine": grad_by_index[row_index]["semantic_grad_align_cosine"],
            "semantic_risk_raw": grad_by_index[row_index]["semantic_risk_raw"],
            "utility_align": grad_by_index[row_index]["utility_grad_align"],
            "utility_align_cosine": grad_by_index[row_index]["utility_grad_align_cosine"],
            "utility_risk_raw": grad_by_index[row_index]["utility_risk_raw"],
            "proxy_mode": grad_by_index[row_index]["proxy_mode"],
            "proxy_key": grad_by_index[row_index]["proxy_key"],
            "proxy_weight_global": float(args.proxy_weight_global),
            "proxy_weight_syntax": float(args.proxy_weight_syntax),
            "proxy_weight_semantic": float(args.proxy_weight_semantic),
            "proxy_weight_utility": float(args.proxy_weight_utility),
            "syntax_size": grad_by_index[row_index]["syntax_size"],
            "semantic_size": grad_by_index[row_index]["semantic_size"],
            "utility_size": grad_by_index[row_index]["utility_size"],
        }
        updated["attribution_score_raw_global"] = grad_by_index[row_index]["global_risk_raw"]
        updated["attribution_score_raw_syntax"] = grad_by_index[row_index]["syntax_risk_raw"]
        updated["attribution_score_raw_semantic"] = grad_by_index[row_index]["semantic_risk_raw"]
        updated["attribution_score_raw_utility"] = grad_by_index[row_index]["utility_risk_raw"]
        updated["attribution_score_raw"] = grad_by_index[row_index]["risk_raw"]
        updated["attribution_score"] = norm_score
        output_rows.append(updated)

    attribution_scores = [row["attribution_score"] for row in output_rows]
    log(f"Saving {len(output_rows)} rows to {args.output_path}")
    save_jsonl(output_rows, args.output_path)
    log(
        "Done. "
        f"attribution_score_range=({min(attribution_scores):.6f}, {max(attribution_scores):.6f})"
    )


if __name__ == "__main__":
    main()
