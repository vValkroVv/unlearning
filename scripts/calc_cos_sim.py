#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

from sentence_transformers import SentenceTransformer


def _load_sbert(device: str) -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def _compute_metric(metric_block: Dict[str, object], model: SentenceTransformer) -> Dict[str, object]:
    value_by_index = metric_block.get("value_by_index", {})
    texts_gt, texts_gen, valid_indices = [], [], []
    for idx, item in value_by_index.items():
        if not isinstance(item, dict):
            continue
        gt = item.get("ground_truth")
        gen = item.get("generation")
        if not gt or not gen:
            print(f"[cos_sim] WARNING: skipping index {idx} — missing ground_truth={gt!r} generation={gen!r}")
            continue
        texts_gt.append(str(gt))
        texts_gen.append(str(gen))
        valid_indices.append(idx)

    if not valid_indices:
        return {"agg_value": 0.0, "value_by_index": {}}

    embs_gt = model.encode(texts_gt, normalize_embeddings=True)
    embs_gen = model.encode(texts_gen, normalize_embeddings=True)
    sims = (embs_gt * embs_gen).sum(axis=1).tolist()

    out_by_index: Dict[str, object] = {}
    for idx, gt, gen, sim in zip(valid_indices, texts_gt, texts_gen, sims):
        out_by_index[str(idx)] = {"cos_sim": float(sim), "ground_truth": gt, "generation": gen}

    agg = sum(sims) / len(sims)
    return {"agg_value": float(agg), "value_by_index": out_by_index}


def process_file(path: Path, model: SentenceTransformer) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False

    out: Dict[str, object] = {}
    updated = False
    for key, block in data.items():
        if not isinstance(block, dict) or "value_by_index" not in block:
            continue
        if not any("ground_truth" in item for item in block["value_by_index"].values() if isinstance(item, dict)):
            continue
        out_key = key.replace("_rouge", "_cos_sim", 1)
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

    try:
        from tqdm import tqdm  # type: ignore

        def _iter(items: Iterable[Path], desc: str):
            return tqdm(list(items), desc=desc, unit="file")
    except Exception:
        def _iter(items: Iterable[Path], desc: str):
            print(f"[cos_sim] {desc}")
            return items

    for bench in benches:
        # Process unlearn runs
        unlearn_root = base_dir / "saves" / "unlearn" / bench
        paths = []
        if unlearn_root.exists():
            paths.extend([p for p in unlearn_root.glob("**/evals/DUET_EVAL.json") if "pretrained" not in p.parts])
        
        # Process base (orig) runs
        base_eval_root = base_dir / "saves" / "evals" / f"{bench}_base"
        if base_eval_root.exists():
            paths.extend(list(base_eval_root.glob("**/DUET_EVAL.json")))

        if not paths:
            print(f"[cos_sim] No eval files found for benchmark: {bench}")
            continue

        for path in _iter(paths, f"{bench}: cos_sim"):
            print(f"[cos_sim] Processing {path}")
            if process_file(path, model):
                total += 1
            else:
                print(f"[cos_sim] Skipped (no value_by_index or ground_truth): {path}")
    print(f"Written COS_SIM_EVAL.json for {total} eval folders.")


if __name__ == "__main__":
    main()
