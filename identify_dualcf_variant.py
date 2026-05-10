#!/usr/bin/env python3
"""Identify which known DualCF artifact variant a folder belongs to.

The script compares the four final train artifacts used by
`scripts/dualcf/run_campaign_one_lr.sh` against known SHA-256 fingerprints:

- `dualcf.zip`
- `dualcf_new.zip` and its byte-identical copies
- `dualcf_v2_5_new_rare.zip`
- `dualcf_v2_6.zip`

The input path can point either to the artifact root itself (a directory that
contains `duet/` and `rwku/`) or to a parent directory that contains one or
more such artifact roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


TARGET_FILES = {
    "rare": Path("duet/rare_llama31_8b_v2/dualcf_rare_v2.jsonl"),
    "popular": Path("duet/popular_llama31_8b_v2/dualcf_popular_v2.jsonl"),
    "merged": Path("duet/merged_llama31_8b_v2/dualcf_merged_v2.jsonl"),
    "rwku": Path("rwku/llama31_8b_level2_v2/dualcf_forget_level2_v2.jsonl"),
}

KNOWN_VARIANTS = {
    "dualcf.zip": {
        "rare": "254bfc5bfeb62ef1744d95fea0a3b6bca451ebe4d6960a876dabb50522ba498a",
        "popular": "9ac6d30f402bdfe7d9af81555732be3344d46ee4498480d9cdacdedf91f5c01c",
        "merged": "285cc03b876fc6df5a5add9be827ccb1d1504acb69b9d1e1f60bc8d26c1c98af",
        "rwku": "4a8995ce47261913cb35466172a3871a02829428712720c738ca578c47e27eed",
    },
    "dualcf_new.zip family": {
        "rare": "1fdf83cce70f5b12a98a5b173a9e36aacbc0da1e7ae0b2175c6934b3c222a8e7",
        "popular": "e44c348ebd50564d8ee2fde10b2476c5c967732e0422db965628cc890a35109c",
        "merged": "0c22c964650c41c05332311d1a53711b4a32e41d4b80885afa32e1930f343580",
        "rwku": "919a99f1f43ecd2917ebfea3bf7577ae3e0c1781fa66b4f95dc4bf16d0a764fd",
        "aliases": ["dualcf_new.zip", "dualcf_v2_5.zip", "dualcf_v2_5_new.zip"],
    },
    "dualcf_v2_5_new_rare.zip": {
        "rare": "9bb2ff30651182ef641605b09b3679c662c2bed96f852447f10687f2aafb8f43",
        "popular": "e44c348ebd50564d8ee2fde10b2476c5c967732e0422db965628cc890a35109c",
        "merged": "0c22c964650c41c05332311d1a53711b4a32e41d4b80885afa32e1930f343580",
        "rwku": "919a99f1f43ecd2917ebfea3bf7577ae3e0c1781fa66b4f95dc4bf16d0a764fd",
    },
    "dualcf_v2_6.zip": {
        "rare": "313c357c5ca2f3282f9ff8b5e17a21e40c984af5c8bc1d88fb609c94f089d2ce",
        "popular": "549973387a6c49bcf081e80e56fda661293d847cd3a3be65eca29b73acbd7693",
        "merged": "31e5b172082d5403ab5c01693fd73a790fc4fea610938e5fdfec0ebbaad72597",
        "rwku": "919a99f1f43ecd2917ebfea3bf7577ae3e0c1781fa66b4f95dc4bf16d0a764fd",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Artifact root directory or a parent directory containing one or more artifact roots.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_artifact_root(path: Path) -> bool:
    return all((path / relative_path).is_file() for relative_path in TARGET_FILES.values())


def discover_artifact_roots(search_root: Path) -> list[Path]:
    resolved_root = search_root.resolve()
    candidates: list[Path] = []

    if is_artifact_root(resolved_root):
        candidates.append(resolved_root)

    for rare_file in resolved_root.rglob("dualcf_rare_v2.jsonl"):
        current = rare_file.parent.resolve()
        while True:
            if current == resolved_root or resolved_root in current.parents:
                if is_artifact_root(current):
                    candidates.append(current)
                if current == resolved_root:
                    break
                current = current.parent
                continue
            break

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(candidates):
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def collect_hashes(artifact_root: Path) -> dict[str, str]:
    return {
        split: sha256_file(artifact_root / relative_path)
        for split, relative_path in TARGET_FILES.items()
    }


def classify_hashes(hashes: dict[str, str]) -> tuple[str | None, dict[str, int]]:
    per_variant_matches: dict[str, int] = {}
    matched_variant: str | None = None
    for variant_name, variant_hashes in KNOWN_VARIANTS.items():
        matches = sum(
            hashes[split] == variant_hashes[split]
            for split in TARGET_FILES
        )
        per_variant_matches[variant_name] = matches
        if matches == len(TARGET_FILES):
            matched_variant = variant_name
    return matched_variant, per_variant_matches


def build_result(artifact_root: Path) -> dict[str, object]:
    hashes = collect_hashes(artifact_root)
    variant_name, per_variant_matches = classify_hashes(hashes)
    result: dict[str, object] = {
        "artifact_root": str(artifact_root),
        "hashes": hashes,
        "variant": variant_name or "unknown",
        "variant_match_counts": per_variant_matches,
    }
    if variant_name == "dualcf_new.zip family":
        result["aliases"] = KNOWN_VARIANTS[variant_name]["aliases"]
    return result


def print_text(results: list[dict[str, object]]) -> None:
    for index, result in enumerate(results, start=1):
        if len(results) > 1:
            print(f"[{index}] artifact_root={result['artifact_root']}")
        else:
            print(f"artifact_root={result['artifact_root']}")

        print(f"variant={result['variant']}")
        aliases = result.get("aliases")
        if aliases:
            print(f"aliases={', '.join(aliases)}")
        print("hashes=")
        for split in ("rare", "popular", "merged", "rwku"):
            print(f"  {split}: {result['hashes'][split]}")

        if result["variant"] == "unknown":
            print("closest_matches=")
            match_counts = result["variant_match_counts"]
            for variant_name, count in sorted(
                match_counts.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"  {variant_name}: {count}/4")

        if index != len(results):
            print()


def main() -> int:
    args = parse_args()
    search_root = args.path.expanduser()
    if not search_root.exists():
        raise FileNotFoundError(f"Path not found: {search_root}")
    if not search_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {search_root}")

    artifact_roots = discover_artifact_roots(search_root)
    if not artifact_roots:
        print(
            "No artifact roots found. Expected a directory containing the four final train files "
            "under duet/ and rwku/.",
            file=sys.stderr,
        )
        return 1

    results = [build_result(artifact_root) for artifact_root in artifact_roots]

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        print_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
