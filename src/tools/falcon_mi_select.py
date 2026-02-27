#!/usr/bin/env python3
"""
MI-based layer selection utility for FALCON.

Implements the paper-style workflow:
- Per-layer mutual information (MI) between forget and retain activations.
- Multi-domain aggregate objective:
    I(l) = sum_i I(F_i^l ; R^l) + eta * sum_{i<j} I(F_i^l ; F_j^l)
- PCA dimensionality reduction (variance threshold) before KDE.
- Gaussian KDE entropy estimate with Scott bandwidth.

Usage examples:
  DUET (multi-domain forget):
    python src/tools/falcon_mi_select.py \
      --model_cfg configs/model/Llama-3.1-8B-Instruct.yaml \
      --model_path /path/to/base_or_sft_checkpoint \
      --dataset_path SwetieePawsss/DUET \
      --forget_splits city_forget_rare_5 city_forget_popular_5 \
      --retain_split city_fast_retain_500 \
      --print_layers

  POPQA (single-domain, list answers):
    python src/tools/falcon_mi_select.py \
      --model_cfg configs/model/Llama-3.1-8B-Instruct.yaml \
      --model_path /path/to/base_or_sft_checkpoint \
      --dataset_path SwetieePawsss/exp_UNLamb \
      --forget_splits rare_forget5_sum \
      --retain_split fast_retain_500 \
      --answer_key possible_answers --answer_index 0 \
      --print_layers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Allow running as `python src/tools/falcon_mi_select.py` without PYTHONPATH tweaks.
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.collators import DataCollatorForSupervisedDataset
from data.qa import QADataset, QAAnswerIndexDataset
from data.utils import IGNORE_INDEX
from model import get_model

EPS = 1e-12


def _log_progress(message: str, quiet: bool) -> None:
    """Emit progress logs to stderr while keeping stdout clean for --print_layers."""
    if quiet:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[MI][{ts}] {message}", file=sys.stderr, flush=True)


def _fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _pool_answer_tokens(
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Pool one vector per sample from layer hidden states [B, S, H].
    Prefer answer-token span (labels != IGNORE_INDEX), fallback to attention mask.
    """
    pooled = []
    batch_size = hidden_states.size(0)
    for b in range(batch_size):
        if labels is not None:
            token_mask = labels[b] != IGNORE_INDEX
        elif attention_mask is not None:
            token_mask = attention_mask[b].bool()
        else:
            token_mask = None

        if token_mask is not None and token_mask.any():
            pooled.append(hidden_states[b, token_mask].mean(dim=0))
        elif attention_mask is not None and attention_mask[b].any():
            last_idx = int(attention_mask[b].sum().item()) - 1
            pooled.append(hidden_states[b, max(last_idx, 0)])
        else:
            pooled.append(hidden_states[b, -1])
    return torch.stack(pooled, dim=0)


def _pca_reduce(features: np.ndarray, pca_var: float) -> np.ndarray:
    """
    Reduce feature dimensionality while preserving `pca_var` variance.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={features.shape}")

    n_samples, n_dims = features.shape
    if n_samples < 5 or n_dims <= 1:
        return features

    max_components = min(n_dims, n_samples - 1)
    if max_components < 1:
        return features

    if pca_var >= 1.0:
        n_components = min(int(pca_var), max_components)
    else:
        n_components = float(pca_var)

    reduced = PCA(n_components=n_components, svd_solver="full").fit_transform(features)

    # gaussian_kde requires feature_dim < sample_count for stable covariance inversion.
    if reduced.shape[1] >= reduced.shape[0]:
        reduced = reduced[:, : max(1, reduced.shape[0] - 1)]
    return reduced


def _kde_entropy(features: np.ndarray, seed: int) -> float:
    """
    KDE plug-in entropy estimate:
        H(X) ~= - E[log p_hat(X)]
    with Gaussian KDE using Scott's bandwidth rule.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 5:
        return float("inf")

    if features.shape[1] >= features.shape[0]:
        features = features[:, : max(1, features.shape[0] - 1)]

    rng = np.random.default_rng(seed)
    for jitter in (0.0, 1e-6, 1e-5):
        try:
            values = features
            if jitter > 0.0:
                values = values + jitter * rng.standard_normal(values.shape)
            kde = gaussian_kde(values.T, bw_method="scott")
            probs = kde.evaluate(values.T)
            return float(-np.mean(np.log(probs + EPS)))
        except Exception:
            continue
    return float("inf")


def _mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    pca_var: float,
    seed: int,
) -> float:
    """
    Estimate I(X;Y) via H(X) + H(Y) - H([X,Y]) after PCA.
    """
    if len(x) < 10 or len(y) < 10:
        return float("inf")

    rng = np.random.default_rng(seed)
    n = min(len(x), len(y))
    idx_x = rng.choice(len(x), size=n, replace=False)
    idx_y = rng.choice(len(y), size=n, replace=False)
    x_n = x[idx_x]
    y_n = y[idx_y]

    x_r = _pca_reduce(x_n, pca_var=pca_var)
    y_r = _pca_reduce(y_n, pca_var=pca_var)
    joint_r = _pca_reduce(np.concatenate([x_n, y_n], axis=1), pca_var=pca_var)

    hx = _kde_entropy(x_r, seed=seed + 11)
    hy = _kde_entropy(y_r, seed=seed + 17)
    hxy = _kde_entropy(joint_r, seed=seed + 23)
    if not np.isfinite(hx + hy + hxy):
        return float("inf")
    return float(hx + hy - hxy)


def _make_dataset(
    dataset_path: str,
    split: str,
    tokenizer,
    template_args,
    question_key: str,
    answer_key: str,
    answer_index: Optional[int],
    max_length: int,
):
    base_kwargs = dict(
        hf_args={"path": dataset_path, "split": split},
        template_args=template_args,
        tokenizer=tokenizer,
        question_key=question_key,
        answer_key=answer_key,
        max_length=max_length,
    )
    if answer_index is None:
        return QADataset(**base_kwargs)
    return QAAnswerIndexDataset(answer_index=answer_index, **base_kwargs)


def _resolve_num_layers(model) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine number of hidden layers from model config.")


@torch.inference_mode()
def _collect_layer_representations(
    model,
    dataloader: DataLoader,
    num_layers: int,
    device: str,
    max_examples: int,
) -> Dict[int, np.ndarray]:
    """
    Collect one pooled activation vector per example for each transformer layer.
    Returns: layer_idx -> [N, H] numpy array
    """
    model.eval()
    reps: Dict[int, List[torch.Tensor]] = {layer: [] for layer in range(num_layers)}
    seen = 0

    for batch in dataloader:
        if seen >= max_examples:
            break

        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        if len(hidden_states) < num_layers + 1:
            raise RuntimeError(
                f"Expected >= {num_layers + 1} hidden states, got {len(hidden_states)}."
            )

        labels = model_inputs.get("labels")
        attention_mask = model_inputs.get("attention_mask")
        batch_size = model_inputs["input_ids"].size(0)
        take = min(batch_size, max_examples - seen)

        for layer in range(num_layers):
            pooled = _pool_answer_tokens(
                hidden_states[layer + 1][:take],
                labels[:take] if labels is not None else None,
                attention_mask[:take] if attention_mask is not None else None,
            )
            reps[layer].append(pooled.detach().float().cpu())

        seen += take

    packed = {}
    for layer in range(num_layers):
        if reps[layer]:
            packed[layer] = torch.cat(reps[layer], dim=0).numpy()
        else:
            packed[layer] = np.zeros((0, 1), dtype=np.float32)
    return packed


def _parse_args():
    parser = argparse.ArgumentParser(description="Select FALCON target layers via MI.")
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_subfolder", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--tokenizer_subfolder", default=None)

    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--forget_splits", nargs="+", required=True)
    parser.add_argument("--retain_split", required=True)

    parser.add_argument("--question_key", default="question")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--answer_index", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--pca_var", type=float, default=0.95)
    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log_every_layers",
        type=int,
        default=1,
        help="Emit MI progress log every N layers (disabled by --quiet).",
    )

    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--print_layers", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    run_start = time.perf_counter()
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    _log_progress(
        "Starting MI selection "
        f"(seed={args.seed}, device={args.device}, max_examples={args.max_examples})",
        quiet=args.quiet,
    )

    stage_start = time.perf_counter()
    cfg = OmegaConf.load(args.model_cfg)
    with open_dict(cfg):
        cfg.model_args.pretrained_model_name_or_path = args.model_path
        if args.model_subfolder:
            cfg.model_args.subfolder = args.model_subfolder
        if args.tokenizer_path:
            cfg.tokenizer_args.pretrained_model_name_or_path = args.tokenizer_path
        if args.tokenizer_subfolder:
            cfg.tokenizer_args.subfolder = args.tokenizer_subfolder

    model, tokenizer = get_model(cfg)
    model.to(args.device)
    num_layers = _resolve_num_layers(model)
    _log_progress(
        f"Loaded model/tokenizer with {num_layers} transformer layers in "
        f"{_fmt_secs(time.perf_counter() - stage_start)}.",
        quiet=args.quiet,
    )
    template_args = cfg.template_args

    collator = DataCollatorForSupervisedDataset(tokenizer)

    stage_start = time.perf_counter()
    retain_dataset = _make_dataset(
        dataset_path=args.dataset_path,
        split=args.retain_split,
        tokenizer=tokenizer,
        template_args=template_args,
        question_key=args.question_key,
        answer_key=args.answer_key,
        answer_index=args.answer_index,
        max_length=args.max_length,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        collate_fn=collator,
    )
    retain_reps = _collect_layer_representations(
        model=model,
        dataloader=retain_loader,
        num_layers=num_layers,
        device=args.device,
        max_examples=args.max_examples,
    )
    retain_count = int(retain_reps[0].shape[0]) if 0 in retain_reps else 0
    _log_progress(
        f"Collected retain representations ({retain_count} examples) in "
        f"{_fmt_secs(time.perf_counter() - stage_start)}.",
        quiet=args.quiet,
    )

    forget_subdomain_reps = []
    for split_idx, forget_split in enumerate(args.forget_splits):
        stage_start = time.perf_counter()
        _log_progress(
            f"Collecting forget representations for split "
            f"'{forget_split}' ({split_idx + 1}/{len(args.forget_splits)}).",
            quiet=args.quiet,
        )
        forget_dataset = _make_dataset(
            dataset_path=args.dataset_path,
            split=forget_split,
            tokenizer=tokenizer,
            template_args=template_args,
            question_key=args.question_key,
            answer_key=args.answer_key,
            answer_index=args.answer_index,
            max_length=args.max_length,
        )
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=True,
            collate_fn=collator,
        )
        forget_subdomain_reps.append(
            _collect_layer_representations(
                model=model,
                dataloader=forget_loader,
                num_layers=num_layers,
                device=args.device,
                max_examples=args.max_examples,
            )
        )
        split_count = int(forget_subdomain_reps[-1][0].shape[0]) if 0 in forget_subdomain_reps[-1] else 0
        _log_progress(
            f"Finished split '{forget_split}' ({split_count} examples) in "
            f"{_fmt_secs(time.perf_counter() - stage_start)}.",
            quiet=args.quiet,
        )

    mi_by_layer: Dict[int, float] = {}
    n_sub = len(forget_subdomain_reps)
    mi_start = time.perf_counter()
    log_every = max(1, int(args.log_every_layers))
    _log_progress(
        f"Computing MI scores across {num_layers} layers.",
        quiet=args.quiet,
    )
    for layer in range(num_layers):
        layer_start = time.perf_counter()
        main_term = 0.0
        for i in range(n_sub):
            main_term += _mutual_information(
                forget_subdomain_reps[i][layer],
                retain_reps[layer],
                pca_var=args.pca_var,
                seed=args.seed + 100 * i + layer,
            )

        inter_term = 0.0
        if n_sub > 1:
            for i in range(n_sub):
                for j in range(i + 1, n_sub):
                    inter_term += _mutual_information(
                        forget_subdomain_reps[i][layer],
                        forget_subdomain_reps[j][layer],
                        pca_var=args.pca_var,
                        seed=args.seed + 10000 + 97 * i + 193 * j + layer,
                    )

        mi_by_layer[layer] = float(main_term + args.eta * inter_term)
        layer_idx = layer + 1
        if layer_idx == 1 or layer_idx % log_every == 0 or layer_idx == num_layers:
            elapsed = time.perf_counter() - mi_start
            avg_per_layer = elapsed / layer_idx
            eta = avg_per_layer * (num_layers - layer_idx)
            _log_progress(
                f"Layer {layer}/{num_layers - 1} score={mi_by_layer[layer]:.6f} "
                f"step={_fmt_secs(time.perf_counter() - layer_start)} "
                f"elapsed={_fmt_secs(elapsed)} "
                f"eta={_fmt_secs(eta)}.",
                quiet=args.quiet,
            )

    layer_ranking = sorted(
        mi_by_layer.keys(),
        key=lambda layer: (not np.isfinite(mi_by_layer[layer]), mi_by_layer[layer]),
    )
    topk = max(1, min(int(args.topk), len(layer_ranking)))
    selected_layers = layer_ranking[:topk]

    payload = {
        "selected_layers": selected_layers,
        "mi_by_layer": {str(layer): float(score) for layer, score in mi_by_layer.items()},
        "args": vars(args),
    }

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2)

    if args.print_layers:
        _log_progress(
            f"Completed MI selection in {_fmt_secs(time.perf_counter() - run_start)}. "
            f"Selected layers: {selected_layers}",
            quiet=args.quiet,
        )
        print(" ".join(str(layer) for layer in selected_layers))
        return

    if not args.quiet:
        _log_progress(
            f"Completed MI selection in {_fmt_secs(time.perf_counter() - run_start)}.",
            quiet=args.quiet,
        )
        print(f"Selected layers (top-{topk}): {selected_layers}")
        print("MI scores (lower is better):")
        for layer in layer_ranking:
            print(f"  layer {layer:>2}: {mi_by_layer[layer]:.6f}")


if __name__ == "__main__":
    main()
