#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Iterable


def _load_sbert(device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def _tokenize(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in text.lower().split():
        counts[token] = counts.get(token, 0) + 1
    return counts


def _cosine_from_counts(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _compute_metric(metric_block: Dict[str, object], model) -> Dict[str, object]:
    value_by_index = metric_block.get("value_by_index", {})
    out_by_index: Dict[str, object] = {}
    scores = []
    for idx, item in value_by_index.items():
        if not isinstance(item, dict):
            continue
        gt = item.get("ground_truth", "")
        gen = item.get("generation", "")
        if gt is None or gen is None:
            continue
        if model is not None:
            emb = model.encode([str(gt), str(gen)], normalize_embeddings=True)
            sim = float(emb[0] @ emb[1])
        else:
            sim = _cosine_from_counts(_tokenize(str(gt)), _tokenize(str(gen)))
        out_by_index[str(idx)] = {
            "cos_sim": sim,
            "ground_truth": gt,
            "generation": gen,
        }
        scores.append(sim)
    agg = float(sum(scores) / len(scores)) if scores else 0.0
    return {"agg_value": agg, "value_by_index": out_by_index}


def process_file(path: Path, model) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False

    out: Dict[str, object] = {}
    updated = False
    for key, block in data.items():
        if not isinstance(block, dict) or "value_by_index" not in block:
            continue
        if "ground_truth" not in next(iter(block.get("value_by_index", {}).values()), {}):
            continue
        out_key = key.replace("_rouge", "_cos_sim")
        out[out_key] = _compute_metric(block, model)
        updated = True

    if not updated:
        return False

    out_path = path.parent / "COS_SIM_EVAL.json"
    out_path.write_text(json.dumps(out, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="duet,popqa,rwku",
        help="Comma-separated list: duet,popqa,rwku",
    )
    parser.add_argument("--gpu", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    base_dir = args.base_dir
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    total = 0
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[cos_sim] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    device = "cuda" if args.gpu not in (None, "", "-1") else "cpu"
    print(f"[cos_sim] Loading SBERT on {device}")
    model = _load_sbert(device=device)
    if model is None:
        print("[cos_sim] WARNING: sentence-transformers not installed; falling back to token cosine.")

    try:
        from tqdm import tqdm  # type: ignore

        def _iter(items: Iterable[Path], desc: str):
            return tqdm(list(items), desc=desc, unit="file")
    except Exception:
        def _iter(items: Iterable[Path], desc: str):
            print(f"[cos_sim] {desc}")
            return items

    for bench in benches:
        root = base_dir / "saves" / "unlearn" / bench
        if not root.exists():
            print(f"[cos_sim] Skip missing benchmark: {bench}")
            continue
        paths = [p for p in root.glob("**/evals/DUET_EVAL.json") if "pretrained" not in p.parts]
        for path in _iter(paths, f"{bench}: cos_sim"):
            run_dir = path.parent.parent
            algo = run_dir.parent.name
            run_name = run_dir.name
            print(f"[cos_sim] {bench} | {algo} | {run_name}")
            if process_file(path, model):
                total += 1
            else:
                print(f"[cos_sim] Skipped (no value_by_index): {path}")
    print(f"Written COS_SIM_EVAL.json for {total} eval folders.")


if __name__ == "__main__":
    main()
