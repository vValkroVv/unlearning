#!/usr/bin/env python3
"""Patch TOFU HF cache metadata for compatibility with datasets==3.0.1.

Why this is needed:
- Some cached TOFU artifacts store feature type as "_type": "List".
- datasets==3.0.1 expects "Sequence" and raises:
  TypeError: must be called with a dataclass type or instance

This script patches both:
1) dataset_info.json files
2) embedded huggingface schema metadata in *.arrow cache files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.ipc as ipc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-cache",
        type=Path,
        default=None,
        help="Path to HF datasets cache root (contains locuslab___tofu).",
    )
    parser.add_argument(
        "--dataset-dir-name",
        type=str,
        default="locuslab___tofu",
        help="Dataset cache directory name under datasets cache root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes but do not write files.",
    )
    return parser.parse_args()


def infer_datasets_cache_root(arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path

    import os

    if os.getenv("HF_DATASETS_CACHE"):
        return Path(os.environ["HF_DATASETS_CACHE"])
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "datasets"
    raise ValueError(
        "Could not infer datasets cache root. Pass --datasets-cache or set HF_DATASETS_CACHE / HF_HOME."
    )


def patch_json_file(path: Path, dry_run: bool) -> bool:
    txt = path.read_text(encoding="utf-8")
    new_txt = txt.replace('"_type": "List"', '"_type": "Sequence"')
    if new_txt == txt:
        return False
    if not dry_run:
        path.write_text(new_txt, encoding="utf-8")
    return True


def patch_arrow_file(path: Path, dry_run: bool) -> bool:
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
        schema = reader.schema

    meta = dict(schema.metadata or {})
    hf_meta = meta.get(b"huggingface")
    if hf_meta is None:
        return False

    old = hf_meta.decode("utf-8")
    new = old.replace('"_type": "List"', '"_type": "Sequence"')
    if new == old:
        return False

    if dry_run:
        return True

    meta[b"huggingface"] = new.encode("utf-8")
    new_schema = schema.with_metadata(meta)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with ipc.new_stream(tmp_path, new_schema) as writer:
        # Keep physical columns unchanged, update only schema metadata.
        writer.write_table(table.cast(new_schema.remove_metadata()))
    tmp_path.replace(path)
    return True


def main() -> None:
    args = parse_args()
    cache_root = infer_datasets_cache_root(args.datasets_cache)
    dataset_root = cache_root / args.dataset_dir_name

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset cache dir not found: {dataset_root}")

    info_files = sorted(dataset_root.rglob("dataset_info.json"))
    arrow_files = sorted(dataset_root.rglob("*.arrow"))

    patched_info = 0
    patched_arrow = 0

    for p in info_files:
        if patch_json_file(p, dry_run=args.dry_run):
            patched_info += 1

    for p in arrow_files:
        if patch_arrow_file(p, dry_run=args.dry_run):
            patched_arrow += 1

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_info_files_scanned": len(info_files),
        "arrow_files_scanned": len(arrow_files),
        "dataset_info_files_patched": patched_info,
        "arrow_files_patched": patched_arrow,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
