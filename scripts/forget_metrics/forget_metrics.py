#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from contextlib import nullcontext

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import datasets
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import sys
repo_root = None
for parent in Path(__file__).resolve().parents:
    if (parent / "configs").exists() and (parent / "src").exists():
        repo_root = parent
        break
if repo_root is None:
    repo_root = Path(__file__).resolve().parents[1]
repo_src = Path(repo_root) / "src"
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data.utils import preprocess_chat_instance, add_dataset_index
from model import get_model


@dataclass
class Example:
    index: int
    question: str
    answer: str
    candidates: Optional[List[str]]
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


def _load_model_cfg(name: str) -> Dict:
    cfg_path = Path(repo_root) / "configs" / "model" / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _load_dataset(benchmark: str, split: str) -> Iterable[Dict]:
    if benchmark == "duet":
        ds = datasets.load_dataset("SwetieePawsss/DUET", split=split)
    elif benchmark == "popqa":
        ds = datasets.load_dataset("SwetieePawsss/exp_UNLamb", split=split)
    elif benchmark == "rwku":
        ds = datasets.load_dataset("SwetieePawsss/exp_r", name=split, split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    ds = add_dataset_index(ds)
    return ds


def _get_fields(benchmark: str) -> Tuple[str, str, Optional[str]]:
    if benchmark == "duet":
        return "question", "answer", None
    if benchmark == "popqa":
        return "question", "possible_answers", "possible_answers"
    if benchmark == "rwku":
        return "query", "answer", None
    raise ValueError(f"Unknown benchmark: {benchmark}")


def _build_examples(
    benchmark: str,
    split: str,
    tokenizer,
    template_args: Dict,
    max_length: int,
) -> List[Example]:
    q_key, a_key, cand_key = _get_fields(benchmark)
    rows = _load_dataset(benchmark, split)
    examples: List[Example] = []
    for row in rows:
        question = row[q_key]
        answer = row[a_key]
        candidates = None
        if cand_key and isinstance(row.get(cand_key), list):
            candidates = list(row[cand_key])
        if isinstance(answer, list):
            # Use first answer for scoring
            answer = answer[0] if answer else ""
        tokenized = preprocess_chat_instance(
            tokenizer,
            template_args,
            [question],
            [answer],
            max_length,
            predict_with_generate=False,
        )
        examples.append(
            Example(
                index=int(row["index"]),
                question=question,
                answer=str(answer),
                candidates=candidates,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                attention_mask=tokenized["attention_mask"],
            )
        )
    return examples


def _collate(batch: List[Example]) -> Dict[str, torch.Tensor]:
    input_ids = [ex.input_ids if isinstance(ex.input_ids, torch.Tensor) else torch.tensor(ex.input_ids, dtype=torch.long) for ex in batch]
    labels = [ex.labels if isinstance(ex.labels, torch.Tensor) else torch.tensor(ex.labels, dtype=torch.long) for ex in batch]
    attention_mask = [ex.attention_mask if isinstance(ex.attention_mask, torch.Tensor) else torch.tensor(ex.attention_mask, dtype=torch.long) for ex in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "indices": torch.tensor([ex.index for ex in batch], dtype=torch.long),
        "questions": [ex.question for ex in batch],
        "answers": [ex.answer for ex in batch],
        "candidates": [ex.candidates for ex in batch],
    }


def _logprob_and_hidden(
    model,
    batch: Dict[str, torch.Tensor],
    need_hidden: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=need_hidden,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels_shift = labels[:, 1:]
    mask = labels_shift != -100
    labels_gather = labels_shift.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, labels_gather.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    seq_logprob = token_logprobs.sum(dim=1)

    # hidden state embedding: mean over answer tokens from last 4 layers
    pooled = None
    if need_hidden:
        hidden_states = outputs.hidden_states[-4:]
        stacked = torch.stack(hidden_states, dim=0).mean(dim=0)  # [B, T, H]
        mask_f = mask.float()
        mask_f = torch.nn.functional.pad(mask_f, (1, 0), value=0.0)  # align with hidden length
        denom = mask_f.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = (stacked * mask_f.unsqueeze(-1)).sum(dim=1) / denom
    return seq_logprob, pooled, log_probs


def _kl_div(base_log_probs: torch.Tensor, model_log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_shift = labels[:, 1:]
    mask = labels_shift != -100
    p = base_log_probs.exp()
    kl = (p * (base_log_probs - model_log_probs)).sum(dim=-1)
    kl = kl * mask
    denom = mask.sum(dim=1).clamp_min(1)
    return kl.sum(dim=1) / denom


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def _score_candidates(
    model,
    tokenizer,
    template_args: Dict,
    question: str,
    candidates: List[str],
    max_length: int,
    amp_dtype: Optional[torch.dtype],
) -> List[float]:
    tokenized_rows = []
    for cand in candidates:
        tokenized_rows.append(
            preprocess_chat_instance(
                tokenizer,
                template_args,
                [question],
                [cand],
                max_length,
                predict_with_generate=False,
            )
        )
    input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in tokenized_rows]
    attention_mask = [torch.tensor(row["attention_mask"], dtype=torch.long) for row in tokenized_rows]
    labels = [torch.tensor(row["labels"], dtype=torch.long) for row in tokenized_rows]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(model.device)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(model.device)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(model.device)

    amp_ctx = nullcontext()
    if amp_dtype is not None and model.device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    with torch.inference_mode(), amp_ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        labels_shift = labels[:, 1:]
        mask = labels_shift != -100
        labels_gather = labels_shift.masked_fill(~mask, 0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, labels_gather.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * mask
        return [float(val.item()) for val in token_logprobs.sum(dim=1)]


def evaluate(
    benchmark: str,
    split: str,
    model_cfg_name: str,
    base_model_path: str,
    adapter_path: Optional[str],
    max_length: int,
    batch_size: int,
    amp_dtype: Optional[torch.dtype],
    compute_hidden: bool,
    compute_kl: bool,
    compute_rank: bool,
    max_examples: Optional[int],
    num_workers: int,
    prefetch_factor: Optional[int],
) -> Tuple[List[Dict], Dict[str, float]]:
    print(f"[forget_metrics] loading model config: {model_cfg_name}")
    model_cfg = _load_model_cfg(model_cfg_name)
    if torch.cuda.is_available():
        if hasattr(model_cfg, "model_args"):
            model_cfg.model_args.device_map = f"cuda:{torch.cuda.current_device()}"
            if getattr(model_cfg.model_args, "low_cpu_mem_usage", None) is None:
                model_cfg.model_args.low_cpu_mem_usage = True
    print(f"[forget_metrics] device_map={getattr(model_cfg.model_args, 'device_map', None)}")
    tokenizer_args = model_cfg.tokenizer_args
    template_args = model_cfg.template_args

    base_cfg = OmegaConf.merge(model_cfg, {})
    base_cfg.model_args.pretrained_model_name_or_path = base_model_path
    base_cfg.model_args.base_model_name_or_path = None

    print(f"[forget_metrics] loading base model: {base_model_path}")
    base_model, tokenizer = get_model(base_cfg)
    base_model.eval()
    device = next(base_model.parameters()).device

    if adapter_path:
        print(f"[forget_metrics] loading adapter: {adapter_path}")
        model_cfg2 = OmegaConf.merge(model_cfg, {})
        model_cfg2.model_args.pretrained_model_name_or_path = adapter_path
        model_cfg2.model_args.base_model_name_or_path = base_model_path
        model, _ = get_model(model_cfg2)
    else:
        model = base_model
    model.eval()

    print(f"[forget_metrics] building examples: benchmark={benchmark} split={split}")
    examples = _build_examples(benchmark, split, tokenizer, template_args, max_length)
    if max_examples is not None:
        examples = examples[:max_examples]
    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    rows: List[Dict] = []
    amp_ctx = nullcontext()
    if amp_dtype is not None and device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    for batch in tqdm(loader, desc=f"[forget_metrics] batches ({benchmark}/{split})", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.inference_mode(), amp_ctx:
            base_logprob, base_hidden, base_log_probs = _logprob_and_hidden(base_model, batch, compute_hidden)
            if adapter_path:
                model_logprob, model_hidden, model_log_probs = _logprob_and_hidden(model, batch, compute_hidden)
                if compute_kl:
                    kl = _kl_div(base_log_probs, model_log_probs, batch["labels"])
                else:
                    kl = torch.full_like(base_logprob, float("nan"))
                if compute_hidden and base_hidden is not None and model_hidden is not None:
                    cos = _cosine(base_hidden, model_hidden)
                else:
                    cos = torch.full_like(base_logprob, float("nan"))
            else:
                model_logprob = base_logprob
                model_hidden = base_hidden
                kl = torch.full_like(base_logprob, float("nan"))
                cos = torch.full_like(base_logprob, float("nan"))

        for i in range(len(batch["indices"])):
            candidates = batch["candidates"][i]
            rank_base = 1
            rank_model = 1
            if compute_rank and candidates and len(candidates) > 1:
                scores_base = _score_candidates(
                    base_model,
                    tokenizer,
                    template_args,
                    batch["questions"][i],
                    candidates,
                    max_length,
                    amp_dtype,
                )
                scores_model = _score_candidates(
                    model,
                    tokenizer,
                    template_args,
                    batch["questions"][i],
                    candidates,
                    max_length,
                    amp_dtype,
                )
                correct = candidates.index(batch["answers"][i]) if batch["answers"][i] in candidates else 0
                rank_base = 1 + sum(s > scores_base[correct] for s in scores_base)
                rank_model = 1 + sum(s > scores_model[correct] for s in scores_model)
            elif not compute_rank:
                rank_base = float("nan")
                rank_model = float("nan")

            rows.append(
                {
                    "index": int(batch["indices"][i].item()),
                    "logprob_base": float(base_logprob[i].item()),
                    "logprob_model": float(model_logprob[i].item()),
                    "delta_logprob": float(model_logprob[i].item() - base_logprob[i].item()),
                    "rank_base": float(rank_base),
                    "rank_model": float(rank_model),
                    "delta_rank": float(rank_model - rank_base),
                    "hidden_cos": float(cos[i].item()),
                    "kl": float(kl[i].item()),
                }
            )

    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and math.isfinite(r[key])]
        return float(sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "logprob_base_mean": _mean("logprob_base"),
        "logprob_model_mean": _mean("logprob_model"),
        "delta_logprob_mean": _mean("delta_logprob"),
        "rank_base_mean": _mean("rank_base"),
        "rank_model_mean": _mean("rank_model"),
        "delta_rank_mean": _mean("delta_rank"),
        "hidden_cos_mean": _mean("hidden_cos"),
        "kl_mean": _mean("kl"),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["duet", "popqa", "rwku"])
    parser.add_argument("--forget-split", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no-hidden", action="store_true")
    parser.add_argument("--no-kl", action="store_true")
    parser.add_argument("--no-rank", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        print(f"[forget_metrics] using cuda:{args.gpu}")
    else:
        print("[forget_metrics] using cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "FORGET_METRICS_SUMMARY.json"
    detail_path = output_dir / "FORGET_METRICS.json"
    if summary_path.exists() and detail_path.exists():
        print(f"[forget_metrics] found existing outputs, skipping: {summary_path}")
        return

    results = {}
    summary = {}
    for split_name, split in [("forget", args.forget_split), ("retain", args.retain_split)]:
        print(f"[forget_metrics] scoring {split_name} split: {split}")
        amp_dtype = None
        if args.amp != "none" and torch.cuda.is_available():
            if args.amp == "bf16" and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            elif args.amp == "fp16":
                amp_dtype = torch.float16
        rows, summ = evaluate(
            benchmark=args.benchmark,
            split=split,
            model_cfg_name=args.model_config,
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            amp_dtype=amp_dtype,
            compute_hidden=not args.no_hidden,
            compute_kl=not args.no_kl,
            compute_rank=not args.no_rank,
            max_examples=args.max_examples,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        results[split_name] = rows
        summary[split_name] = summ
        print(f"[forget_metrics] {split_name} summary: {summ}")

    (output_dir / "FORGET_METRICS.json").write_text(json.dumps(results, indent=2))
    (output_dir / "FORGET_METRICS_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"[forget_metrics] wrote {output_dir / 'FORGET_METRICS.json'}")
    print(f"[forget_metrics] wrote {output_dir / 'FORGET_METRICS_SUMMARY.json'}")


if __name__ == "__main__":
    main()
