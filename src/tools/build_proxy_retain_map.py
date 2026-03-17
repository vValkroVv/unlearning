#!/usr/bin/env python3
"""Build syntax, semantic, and utility proxy retain banks for DualCF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    delex_template,
    load_dataset_split,
    save_jsonl,
    tokenize_normalized_words,
)


def log(message: str) -> None:
    print(f"[build_proxy_retain_map] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forget-dataset-path", required=True)
    parser.add_argument("--forget-split", required=True)
    parser.add_argument("--forget-dataset-name", default=None)
    parser.add_argument("--forget-data-files", default=None)
    parser.add_argument("--retain-dataset-path", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--retain-dataset-name", default=None)
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--forget-question-key", default="question")
    parser.add_argument("--retain-question-key", default=None)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--fallback-top-k", type=int, default=8)
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument("--utility-top-k", type=int, default=8)
    parser.add_argument("--embed-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--utility-dataset-path", default=None)
    parser.add_argument("--utility-split", default="train")
    parser.add_argument("--utility-dataset-name", default=None)
    parser.add_argument("--utility-data-files", default=None)
    parser.add_argument("--utility-question-key", default="question")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def jaccard_score(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    if not left or not right:
        return 0.0
    return float(len(left & right)) / float(len(left | right))


@torch.inference_mode()
def encode_texts(texts: Sequence[str], model_name: str, batch_size: int, device: str) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        rows.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(rows, dim=0)


def main():
    args = parse_args()
    retain_question_key = args.retain_question_key or args.forget_question_key
    forget_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.forget_dataset_path,
            split=args.forget_split,
            name=args.forget_dataset_name,
            data_files=args.forget_data_files,
            max_examples=args.max_examples,
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
    utility_rows: list[dict] = []
    if args.utility_dataset_path not in (None, "", "null", "None"):
        utility_rows = [
            dict(row)
            for row in load_dataset_split(
                path=args.utility_dataset_path,
                split=args.utility_split,
                name=args.utility_dataset_name,
                data_files=args.utility_data_files,
            )
        ]
    log(
        f"Loaded forget_rows={len(forget_rows)} retain_rows={len(retain_rows)} "
        f"utility_rows={len(utility_rows)}"
    )

    retain_by_template: Dict[str, List[int]] = {}
    retain_tokens: List[Tuple[str, List[str]]] = []
    for row in retain_rows:
        template_key = delex_template(row[retain_question_key])
        retain_by_template.setdefault(template_key, []).append(int(row["index"]))
    for template_key in retain_by_template:
        retain_tokens.append((template_key, tokenize_normalized_words(template_key)))

    forget_questions = [str(row[args.forget_question_key]) for row in forget_rows]
    retain_questions = [str(row[retain_question_key]) for row in retain_rows]
    forget_emb = encode_texts(
        forget_questions,
        model_name=args.embed_model_name,
        batch_size=args.embed_batch_size,
        device=args.embed_device,
    )
    retain_emb = encode_texts(
        retain_questions,
        model_name=args.embed_model_name,
        batch_size=args.embed_batch_size,
        device=args.embed_device,
    )
    utility_emb = None
    if utility_rows:
        utility_questions = [str(row[args.utility_question_key]) for row in utility_rows]
        utility_emb = encode_texts(
            utility_questions,
            model_name=args.embed_model_name,
            batch_size=args.embed_batch_size,
            device=args.embed_device,
        )

    output_rows = []
    exact_matches = 0
    fallback_matches = 0
    syntax_sizes: List[int] = []
    semantic_sizes: List[int] = []
    utility_sizes: List[int] = []

    for row_idx, row in enumerate(forget_rows):
        template_key = delex_template(row[args.forget_question_key])
        syntax_retain_indices = list(retain_by_template.get(template_key, []))
        proxy_mode = "template_exact"
        if syntax_retain_indices:
            exact_matches += 1
            syntax_retain_indices = syntax_retain_indices[: int(args.top_k)]
        else:
            proxy_mode = "template_fallback"
            forget_tokens = tokenize_normalized_words(template_key)
            scored = []
            for retain_template, retain_template_tokens in retain_tokens:
                scored.append(
                    (
                        jaccard_score(forget_tokens, retain_template_tokens),
                        retain_template,
                    )
                )
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_templates = [
                template for _, template in scored[: int(args.fallback_top_k)] if template
            ]
            syntax_retain_indices = []
            for selected_template in selected_templates:
                syntax_retain_indices.extend(retain_by_template.get(selected_template, []))
            syntax_retain_indices = syntax_retain_indices[: int(args.top_k)]
            fallback_matches += 1

        semantic_retain_indices: list[int] = []
        if len(retain_rows) > 0:
            sim_ret = torch.mv(retain_emb, forget_emb[row_idx])
            sem_k = min(int(args.semantic_top_k), sim_ret.numel())
            if sem_k > 0:
                sem_pos = torch.topk(sim_ret, k=sem_k).indices.tolist()
                semantic_retain_indices = [int(retain_rows[pos]["index"]) for pos in sem_pos]

        utility_anchor_indices: list[int] = []
        if utility_emb is not None and utility_rows:
            sim_util = torch.mv(utility_emb, forget_emb[row_idx])
            util_k = min(int(args.utility_top_k), sim_util.numel())
            if util_k > 0:
                util_pos = torch.topk(sim_util, k=util_k).indices.tolist()
                utility_anchor_indices = [int(utility_rows[pos]["index"]) for pos in util_pos]

        syntax_sizes.append(len(syntax_retain_indices))
        semantic_sizes.append(len(semantic_retain_indices))
        utility_sizes.append(len(utility_anchor_indices))
        output_rows.append(
            {
                "index": int(row["index"]),
                "template_key": template_key,
                "proxy_mode": proxy_mode,
                "retain_indices": syntax_retain_indices,
                "syntax_retain_indices": syntax_retain_indices,
                "semantic_retain_indices": semantic_retain_indices,
                "utility_anchor_indices": utility_anchor_indices,
            }
        )

    save_jsonl(output_rows, args.output_path)
    log(f"Saved proxy map rows={len(output_rows)} path={args.output_path}")

    if args.sidecar_path:
        sidecar = {
            "forget_rows": len(forget_rows),
            "retain_rows": len(retain_rows),
            "utility_rows": len(utility_rows),
            "exact_matches": exact_matches,
            "fallback_matches": fallback_matches,
            "syntax_size_min": min(syntax_sizes) if syntax_sizes else 0,
            "syntax_size_max": max(syntax_sizes) if syntax_sizes else 0,
            "syntax_size_mean": sum(syntax_sizes) / float(len(syntax_sizes)) if syntax_sizes else 0.0,
            "semantic_size_min": min(semantic_sizes) if semantic_sizes else 0,
            "semantic_size_max": max(semantic_sizes) if semantic_sizes else 0,
            "semantic_size_mean": sum(semantic_sizes) / float(len(semantic_sizes)) if semantic_sizes else 0.0,
            "utility_size_min": min(utility_sizes) if utility_sizes else 0,
            "utility_size_max": max(utility_sizes) if utility_sizes else 0,
            "utility_size_mean": sum(utility_sizes) / float(len(utility_sizes)) if utility_sizes else 0.0,
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar to {args.sidecar_path}")


if __name__ == "__main__":
    main()
