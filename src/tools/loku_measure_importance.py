#!/usr/bin/env python3
"""Measure LoKU FILA importance tensors on forget/retain batches.

Produces an artifact with keys:
- importance_f: Dict[name -> Tensor]
- importance_r: Dict[name -> Tensor]
- f_cnt: int
- r_cnt: int
- target_modules: List[str]
- meta: Dict
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import get_collators, get_data
from data.utils import IGNORE_INDEX
from model import get_model
from model.fila import collect_fila_target_parameters
from trainer.utils import _filter_model_inputs, seed_everything


def _log(msg: str, quiet: bool) -> None:
    if quiet:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LoKU-IMP][{ts}] {msg}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure LoKU FILA importance tensors.")
    parser.add_argument("--config-name", default="unlearn.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides. Prefix with '--' before first override.",
    )
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
    overrides = [f"experiment={args.experiment}"]
    extra = list(args.overrides)
    if extra and extra[0] == "--":
        extra = extra[1:]
    overrides.extend(extra)

    config_dir = str(SRC_ROOT.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    return cfg, extra


def _resolve_target_modules(cfg) -> List[str]:
    lora_cfg = cfg.model.get("lora_config", None)
    if lora_cfg is None:
        raise ValueError("LoKU importance measurement requires a LoRA model config.")
    target_modules = list(lora_cfg.get("target_modules", []) or [])
    if not target_modules:
        raise ValueError("model.lora_config.target_modules must be non-empty.")
    if any(str(name).split(".")[-1] == "lm_head" for name in target_modules):
        raise ValueError(
            "LoKU importance measurement requires `lm_head` to be excluded from "
            "model.lora_config.target_modules."
        )
    return target_modules


def _select_device(args: argparse.Namespace) -> str:
    if args.device:
        return str(args.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_model_and_data(cfg, args: argparse.Namespace):
    model_cfg = cfg.model
    template_args = model_cfg.template_args

    model, tokenizer = get_model(model_cfg)
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False
    device = _select_device(args)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)

    mode = cfg.get("mode", "unlearn")
    data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args)
    if "train" not in data:
        raise ValueError("Expected `train` split in unlearn data pipeline.")

    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    bs = args.batch_size
    if bs is None:
        bs = int(cfg.trainer.args.get("per_device_train_batch_size", 1))

    dataloader = DataLoader(
        data["train"],
        batch_size=max(1, int(bs)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collator,
    )
    return model, tokenizer, dataloader, device


def _to_model_inputs(batch_split, device: str) -> Dict[str, torch.Tensor]:
    if isinstance(batch_split, dict) and "original" in batch_split:
        batch_split = batch_split["original"]
    model_inputs = _filter_model_inputs(batch_split)
    if "labels" not in model_inputs:
        raise ValueError("Batch is missing labels required for CE-based importance.")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _accumulate_for_split(
    model,
    model_inputs: Dict[str, torch.Tensor],
    target_params: Dict[str, torch.nn.Parameter],
    accumulator: Dict[str, torch.Tensor],
) -> int:
    labels = model_inputs["labels"]
    token_count = int((labels != IGNORE_INDEX).sum().item())
    if token_count <= 0:
        return 0

    model.zero_grad(set_to_none=True)
    outputs = model(**model_inputs)
    outputs.loss.backward()

    scale = float(token_count)
    for name, param in target_params.items():
        grad = param.grad
        if grad is None:
            continue
        accumulator[name] += (grad.detach().float().pow(2) * scale).cpu()

    model.zero_grad(set_to_none=True)
    return token_count


def _init_accumulators(
    target_params: Dict[str, torch.nn.Parameter],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    importance_f = {
        name: torch.zeros_like(param.detach().float(), device="cpu")
        for name, param in target_params.items()
    }
    importance_r = {
        name: torch.zeros_like(param.detach().float(), device="cpu")
        for name, param in target_params.items()
    }
    return importance_f, importance_r


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    cfg, user_overrides = _compose_cfg(args)

    # Optional CLI batch-size override without requiring a Hydra override.
    if args.batch_size is not None:
        with open_dict(cfg):
            cfg.trainer.args.per_device_train_batch_size = int(args.batch_size)

    target_modules = _resolve_target_modules(cfg)
    _log(f"Target modules: {target_modules}", quiet=args.quiet)

    model, _, dataloader, device = _prepare_model_and_data(cfg, args)

    target_params = collect_fila_target_parameters(
        peft_model=model,
        target_modules=target_modules,
    )
    if not target_params:
        raise ValueError(
            "No LoRA base-layer target weights found. Ensure model is LoRA-wrapped and "
            "target_modules match injected adapters."
        )

    # For importance measurement we only need gradients on FILA-target base weights.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in target_params.values():
        param.requires_grad_(True)

    model.train()

    importance_f, importance_r = _init_accumulators(target_params)
    f_cnt = 0
    r_cnt = 0
    steps = 0

    max_steps = int(args.max_steps)
    for batch in dataloader:
        if max_steps > 0 and steps >= max_steps:
            break

        if "forget" not in batch or "retain" not in batch:
            raise ValueError("Expected batch with `forget` and `retain` keys.")

        forget_inputs = _to_model_inputs(batch["forget"], device=device)
        retain_inputs = _to_model_inputs(batch["retain"], device=device)

        f_cnt += _accumulate_for_split(
            model=model,
            model_inputs=forget_inputs,
            target_params=target_params,
            accumulator=importance_f,
        )
        r_cnt += _accumulate_for_split(
            model=model,
            model_inputs=retain_inputs,
            target_params=target_params,
            accumulator=importance_r,
        )

        steps += 1
        if not args.quiet and steps % 10 == 0:
            _log(
                f"Processed {steps} steps (f_cnt={f_cnt}, r_cnt={r_cnt}).",
                quiet=False,
            )

    if f_cnt <= 0 or r_cnt <= 0:
        raise ValueError(
            f"Invalid counters after measurement: f_cnt={f_cnt}, r_cnt={r_cnt}."
        )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    payload = {
        "importance_f": importance_f,
        "importance_r": importance_r,
        "f_cnt": int(f_cnt),
        "r_cnt": int(r_cnt),
        "target_modules": list(target_modules),
        "meta": {
            "config_name": args.config_name,
            "experiment": args.experiment,
            "overrides": user_overrides,
            "steps": int(steps),
            "max_steps": int(max_steps),
            "batch_size": int(cfg.trainer.args.per_device_train_batch_size),
            "seed": int(args.seed),
            "device": device,
        },
    }
    torch.save(payload, args.output_path)

    _log(
        f"Saved importance file to {args.output_path} "
        f"(layers={len(importance_f)}, f_cnt={f_cnt}, r_cnt={r_cnt}, steps={steps})",
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
