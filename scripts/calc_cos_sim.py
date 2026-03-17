#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


def _candidate_snapshot_dir(model_id: str, cache_root: Path) -> Path | None:
    repo_dir = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def _resolve_sbert_model_path(model_ref: str) -> str:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    snapshot_dir = _candidate_snapshot_dir(model_ref, hf_home)
    if snapshot_dir is not None:
        return str(snapshot_dir.resolve())

    try:
        resolved = snapshot_download(
            repo_id=model_ref,
            cache_dir=str(hf_home),
            local_files_only=offline,
        )
        return str(Path(resolved).resolve())
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve SBERT model '{model_ref}' from local path or HF cache under {hf_home}. "
            "If you are on the offline GPU box, make sure the model is cached in HF_HOME/hub "
            "or pass --sbert_model_path explicitly."
        ) from exc


def _load_sbert(device: str, model_ref: str) -> SentenceTransformer:
    resolved_model = _resolve_sbert_model_path(model_ref)
    print(f"[cos_sim] Resolved SBERT model: {resolved_model}")
    return SentenceTransformer(resolved_model, device=device)


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
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    out: Dict[str, object] = {}
    summary_out: Dict[str, float] = {}
    updated = False
    for key, block in data.items():
        if not isinstance(block, dict) or "value_by_index" not in block:
            continue
        if not any("ground_truth" in item for item in block["value_by_index"].values() if isinstance(item, dict)):
            continue
        out_key = key.replace("_rouge", "_cos_sim", 1)
        metric_out = _compute_metric(block, model)
        out[out_key] = metric_out
        summary_out[out_key] = float(metric_out["agg_value"])
        updated = True

    if not updated:
        return False

    out_path = path.parent / "COS_SIM_EVAL.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary_path = path.parent / "COS_SIM_SUMMARY.json"
    summary_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return True


def resolve_search_roots(path_to_saves: Path) -> list[Path]:
    root = path_to_saves.expanduser().resolve()
    candidates: list[Path] = []

    if root.name == "unlearn":
        candidates.append(root)
        eval_root = root.parent / "evals"
        if eval_root.exists():
            candidates.append(eval_root)
    elif root.name == "saves":
        unlearn_root = root / "unlearn"
        eval_root = root / "evals"
        if unlearn_root.exists():
            candidates.append(unlearn_root)
        if eval_root.exists():
            candidates.append(eval_root)
        if not candidates:
            candidates.append(root)
    else:
        if (root / "unlearn").exists():
            candidates.append(root / "unlearn")
        if (root / "evals").exists():
            candidates.append(root / "evals")
        if not candidates:
            candidates.append(root)

    return candidates


def collect_eval_paths(path_to_saves: Path) -> list[Path]:
    paths: set[Path] = set()
    for search_root in resolve_search_roots(path_to_saves):
        if not search_root.exists():
            continue
        for path in search_root.rglob("DUET_EVAL.json"):
            if "pretrained" in path.parts:
                continue
            paths.add(path.resolve())
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_saves", type=Path, required=True, help="Path to saves/ or saves/unlearn")
    parser.add_argument("--gpu", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--sbert_model_path",
        type=str,
        default=os.environ.get("SBERT_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Local path or HF repo id for the SBERT model used for cosine similarity.",
    )
    args = parser.parse_args()

    total = 0
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[cos_sim] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    device = "cuda" if args.gpu not in (None, "", "-1") else "cpu"
    print(f"[cos_sim] Loading SBERT on {device}")
    model = _load_sbert(device=device, model_ref=args.sbert_model_path)

    try:
        from tqdm import tqdm  # type: ignore

        def _iter(items: Iterable[Path], desc: str):
            return tqdm(list(items), desc=desc, unit="file")
    except Exception:
        def _iter(items: Iterable[Path], desc: str):
            print(f"[cos_sim] {desc}")
            return items

    paths = collect_eval_paths(args.path_to_saves)
    if not paths:
        print(f"[cos_sim] No DUET_EVAL.json files found under {args.path_to_saves}")
        print("[cos_sim] Expected to find files like evals/DUET_EVAL.json and checkpoint_evals/checkpoint-*/DUET_EVAL.json")
        return

    print(f"[cos_sim] Found {len(paths)} DUET_EVAL.json files under {args.path_to_saves}")
    for path in _iter(paths, "cos_sim"):
        print(f"[cos_sim] Processing {path}")
        if process_file(path, model):
            total += 1
        else:
            print(f"[cos_sim] Skipped (no value_by_index or ground_truth): {path}")
    print(f"Written COS_SIM_EVAL.json for {total} eval folders.")


if __name__ == "__main__":
    main()
