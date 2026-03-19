#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import load_dataset_split, load_keyed_jsonish, resolve_answer


class CandidateBundle(BaseModel):
    alternates: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    relation_scores: list[float] = Field(default_factory=list)
    shared_fact_scores: list[float] = Field(default_factory=list)
    candidate_sources: list[str] = Field(default_factory=list)
    answer_type: str = "unknown"


@dataclass
class RunMeta:
    created_at_utc: str
    backend: str
    model: str
    prompt_family: str
    dataset_path: str
    dataset_name: Optional[str]
    split: str
    data_files: Optional[str]
    question_key: str
    answer_key: str
    answer_index: Optional[int]
    num_alternates: int
    batch_size: int
    max_examples: int
    candidate_bank: Optional[str]
    candidate_bank_limit: int
    concurrent: int
    reasoning_effort: Optional[str]
    timeout_seconds: float
    sleep_seconds: float
    max_attempts: int
    resume: bool
    codex_login_status: Optional[str] = None


def log(message: str) -> None:
    print(f"[generate_codex_cf_sidecar] {message}", flush=True)


def clean_text(text: Any) -> str:
    value = str(text or "").strip().strip('"').strip("'")
    if value:
        value = value.splitlines()[0].strip()
    value = " ".join(value.split())
    return value[:128]


def normalize_text(text: Any) -> str:
    return " ".join(clean_text(text).lower().split())


def maybe_str(value: Any) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def build_run_meta(
    args: argparse.Namespace,
    *,
    codex_status: Optional[str],
) -> RunMeta:
    return RunMeta(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        backend="codex_cli",
        model=args.model,
        prompt_family=args.prompt_family,
        dataset_path=args.dataset_path,
        dataset_name=maybe_str(args.dataset_name),
        split=args.split,
        data_files=maybe_str(args.data_files),
        question_key=args.question_key,
        answer_key=args.answer_key,
        answer_index=args.answer_index,
        num_alternates=args.num_alternates,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        candidate_bank=maybe_str(args.candidate_bank),
        candidate_bank_limit=args.candidate_bank_limit,
        concurrent=args.concurrent,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        max_attempts=args.max_attempts,
        resume=bool(args.resume),
        codex_login_status=codex_status,
    )


def write_run_meta(
    output_path: Path,
    args: argparse.Namespace,
    *,
    codex_status: Optional[str],
) -> None:
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    meta_path.write_text(
        json.dumps(asdict(build_run_meta(args, codex_status=codex_status)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def model_validate(model_cls: type[BaseModel], payload: Any) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    return model_cls.parse_obj(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a verified multi-candidate DualCF sidecar JSONL with Codex CLI."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--question-key", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--answer-index", type=int, default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-family",
        choices=("default", "strict_short", "duet_relation_safe", "rwku_shared_fact_safe"),
        required=True,
    )
    parser.add_argument("--num-alternates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--candidate-bank-limit", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.concurrent <= 0:
        raise ValueError("--concurrent must be >= 1")


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset_split(
        path=args.dataset_path,
        split=args.split,
        name=maybe_str(args.dataset_name),
        data_files=maybe_str(args.data_files),
        max_examples=args.max_examples,
    )
    rows = [dict(row) for row in dataset]
    for row_no, row in enumerate(rows, start=1):
        if "index" not in row:
            raise KeyError(f"Dataset row_no={row_no} is missing `index`.")
    return rows


def load_existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(int(row["index"]))
    return done


def load_candidate_bank(path: Optional[str]) -> dict[str, dict[str, Any]]:
    if path in (None, "", "null", "None"):
        return {}
    return load_keyed_jsonish(path, key_field="index")


def build_prompt_rules(prompt_family: str) -> list[str]:
    rules = [
        "Keep the same answer type as the gold answer.",
        "Output only short answer spans, not explanations or full sentences.",
        "Never repeat the gold answer.",
        "Never output bullets, numbering, or commentary.",
    ]
    if prompt_family == "duet_relation_safe":
        rules.extend(
            [
                "Preserve the same semantic relation as the gold answer.",
                "Prefer candidate-bank-compatible alternatives when they fit.",
                "Keep answers concise and relation-safe.",
            ]
        )
    elif prompt_family == "rwku_shared_fact_safe":
        rules.extend(
            [
                "Change only the target answer.",
                "Avoid altering unrelated shared or public facts.",
                "Be more conservative than DUET and avoid fabricated explanations.",
            ]
        )
    elif prompt_family == "strict_short":
        rules.extend(
            [
                "Prefer compact one-line spans.",
                "Avoid unnecessary modifiers.",
            ]
        )
    else:
        rules.extend(
            [
                "Keep the answer plausible and concise.",
                "Preserve the question's answer type and scope when possible.",
            ]
        )
    return rules


def build_codex_batch_schema() -> dict[str, Any]:
    score_array = {
        "type": "array",
        "items": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "alternates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                        },
                        "scores": score_array,
                        "relation_scores": score_array,
                        "shared_fact_scores": score_array,
                        "candidate_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "answer_type": {"type": "string"},
                    },
                    "required": [
                        "index",
                        "alternates",
                        "scores",
                        "relation_scores",
                        "shared_fact_scores",
                        "candidate_sources",
                        "answer_type",
                    ],
                },
            }
        },
        "required": ["results"],
    }


def build_codex_batch_prompt(
    *,
    prompt_family: str,
    num_alternates: int,
    batch_rows: list[dict[str, Any]],
) -> str:
    policy_lines = "\n".join(f"- {rule}" for rule in build_prompt_rules(prompt_family))
    batch_payload = []
    for row in batch_rows:
        payload = {
            "index": int(row["index"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
        }
        if row["candidate_answers"]:
            payload["candidate_answers"] = list(row["candidate_answers"])
        batch_payload.append(payload)

    return (
        "You are generating verified multi-alternate counterfactual sidecar rows for DualCF v3.\n\n"
        "Return JSON only matching the provided schema.\n"
        "The top-level object must be {\"results\": [...]}.\n"
        "results must contain exactly one item per input example.\n"
        "Preserve the input index exactly.\n"
        f"Return exactly {num_alternates} short wrong alternatives when possible.\n"
        "Never repeat the gold answer.\n"
        "Never output explanations, labels, or full sentences.\n"
        "For each result, align any metadata arrays with alternates.\n\n"
        "Policy:\n"
        f"{policy_lines}\n\n"
        "Per result:\n"
        "- alternates: short counterfactual answers\n"
        "- scores: overall quality in [0,1]\n"
        "- relation_scores: relation preservation in [0,1]\n"
        "- shared_fact_scores: shared/public fact preservation in [0,1]\n"
        "- candidate_sources: provenance strings aligned to alternates\n"
        "- answer_type: short label like year/date/person/place/number/string/unknown\n\n"
        "Batch:\n"
        + json.dumps(batch_payload, ensure_ascii=False, indent=2)
    )


def _pick_list(values: list[Any], kept_indices: list[int], default_value: Any) -> list[Any]:
    output: list[Any] = []
    for idx in kept_indices:
        if idx < len(values):
            output.append(values[idx])
        else:
            output.append(default_value)
    return output


def _normalize_float_list(values: list[Any], expected_length: int, default_value: float) -> list[float]:
    normalized: list[float] = []
    for raw_value in values[:expected_length]:
        try:
            value = float(raw_value)
        except Exception:
            value = default_value
        value = max(0.0, min(1.0, value))
        normalized.append(value)
    while len(normalized) < expected_length:
        normalized.append(default_value)
    return normalized


def normalize_bundle(
    *,
    bundle: CandidateBundle,
    answer: str,
    num_alternates: int,
    default_source: str,
) -> dict[str, Any]:
    answer_norm = normalize_text(answer)
    raw_alts = [clean_text(item) for item in bundle.alternates]
    deduped_alts: list[str] = []
    seen: set[str] = set()
    kept_indices: list[int] = []

    for idx, alt in enumerate(raw_alts):
        alt_norm = normalize_text(alt)
        if not alt_norm:
            continue
        if alt_norm == answer_norm:
            continue
        if alt_norm in seen:
            continue
        seen.add(alt_norm)
        deduped_alts.append(alt)
        kept_indices.append(idx)
        if len(deduped_alts) >= num_alternates:
            break

    if not deduped_alts:
        raise ValueError("No usable alternates after normalization.")

    scores = _normalize_float_list(
        _pick_list(list(bundle.scores), kept_indices, 0.5),
        len(deduped_alts),
        0.5,
    )
    relation_scores = _normalize_float_list(
        _pick_list(list(bundle.relation_scores), kept_indices, 1.0),
        len(deduped_alts),
        1.0,
    )
    shared_fact_scores = _normalize_float_list(
        _pick_list(list(bundle.shared_fact_scores), kept_indices, 1.0),
        len(deduped_alts),
        1.0,
    )

    raw_sources = _pick_list(list(bundle.candidate_sources), kept_indices, default_source)
    candidate_sources: list[str] = []
    for source in raw_sources:
        source_text = clean_text(source) or default_source
        candidate_sources.append(source_text)
    while len(candidate_sources) < len(deduped_alts):
        candidate_sources.append(default_source)

    answer_type = clean_text(bundle.answer_type) or "unknown"
    return {
        "alternates": deduped_alts,
        "scores": scores,
        "relation_scores": relation_scores,
        "shared_fact_scores": shared_fact_scores,
        "candidate_sources": candidate_sources[: len(deduped_alts)],
        "answer_type": answer_type,
        "candidate_count": len(deduped_alts),
    }


def build_sidecar_row(
    *,
    index: int,
    normalized: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_version = f"codex_cli:{args.prompt_family}:v1"
    return {
        "index": index,
        "alternates": normalized["alternates"],
        "scores": normalized["scores"],
        "relation_scores": normalized["relation_scores"],
        "shared_fact_scores": normalized["shared_fact_scores"],
        "candidate_sources": normalized["candidate_sources"],
        "answer_type": normalized["answer_type"],
        "candidate_count": normalized["candidate_count"],
        "generator": "codex_cli",
        "generator_backend": "codex_cli",
        "generator_model": args.model,
        "generator_reasoning_effort": args.reasoning_effort,
        "model": args.model,
        "prompt_family": args.prompt_family,
        "prompt_version": prompt_version,
        "structured_outputs": True,
    }


def codex_login_status() -> str:
    if shutil.which("codex") is None:
        raise RuntimeError("`codex` CLI is not installed or not on PATH.")

    proc = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    status_text = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "Unable to read Codex CLI login status. "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return status_text


def run_codex_batch(
    *,
    args: argparse.Namespace,
    prompt: str,
    schema_path: Path,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    workspace_dir: Path,
) -> dict[str, Any]:
    cmd = [
        "codex",
        "-s",
        "read-only",
        "-a",
        "never",
        "exec",
        "-C",
        str(workspace_dir),
        "--skip-git-repo-check",
        "--model",
        args.model,
        "--color",
        "never",
        "--output-schema",
        str(schema_path),
        "-o",
        str(result_path),
        "-",
    ]
    if args.reasoning_effort:
        cmd[1:1] = ["-c", f'model_reasoning_effort="{args.reasoning_effort}"']
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if (
            "refresh token was already used" in combined
            or "Please log out and sign in again" in combined
        ):
            raise RuntimeError(
                "Codex CLI authentication is stale. Run `codex logout` and "
                "`codex login` again, then rerun with --resume."
            )
        if "not supported when using Codex with a ChatGPT account" in combined:
            raise RuntimeError(
                f"Model `{args.model}` is not supported for Codex ChatGPT-login auth on this machine. "
                "Use a supported ChatGPT-login model or switch to API-key auth."
            )
        if '"code":"model_not_found"' in combined or "model_not_found" in combined:
            raise RuntimeError(f"Codex model `{args.model}` was not found.")
        raise RuntimeError(
            f"codex exec failed with code {proc.returncode}; see {stdout_path} and {stderr_path}"
        )

    if not result_path.exists():
        raise RuntimeError(f"Codex did not write {result_path}")

    return json.loads(result_path.read_text(encoding="utf-8"))


def generate_codex_sidecar(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    candidate_bank: dict[str, dict[str, Any]],
    output_path: Path,
    done_indices: set[int],
) -> tuple[int, str]:
    status_text = codex_login_status()
    pending_rows: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["index"])
        if idx in done_indices:
            continue
        answer = resolve_answer(
            row=row,
            answer_key=args.answer_key,
            answer_index=args.answer_index,
        )
        bank_row = candidate_bank.get(str(idx), {})
        pending_rows.append(
            {
                "index": idx,
                "question": clean_text(row.get(args.question_key, "")),
                "answer": answer,
                "candidate_answers": [
                    clean_text(item)
                    for item in bank_row.get("candidate_answers", [])[: args.candidate_bank_limit]
                    if clean_text(item)
                ],
            }
        )

    debug_root = output_path.parent / "_codex_batches"
    workspace_dir = output_path.parent / "_codex_workspace"
    debug_root.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    schema_path = debug_root / "schema.json"
    schema_path.write_text(
        json.dumps(build_codex_batch_schema(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    default_source = f"codex_cli:{args.model}"
    written = 0
    batch_starts = list(range(0, len(pending_rows), args.batch_size))
    progress_desc = f"{args.split}:{args.model}"
    if args.reasoning_effort:
        progress_desc += f":{args.reasoning_effort}"

    def run_one_batch(batch_id: int, start: int) -> tuple[int, list[str]]:
        batch = pending_rows[start : start + args.batch_size]
        prompt = build_codex_batch_prompt(
            prompt_family=args.prompt_family,
            num_alternates=args.num_alternates,
            batch_rows=batch,
        )
        prompt_path = debug_root / f"batch_{batch_id:05d}.prompt.txt"
        result_path = debug_root / f"batch_{batch_id:05d}.result.json"
        stdout_path = debug_root / f"batch_{batch_id:05d}.stdout.log"
        stderr_path = debug_root / f"batch_{batch_id:05d}.stderr.log"
        batch_workspace_dir = workspace_dir / f"batch_{batch_id:05d}"
        batch_workspace_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        last_error: Optional[Exception] = None
        payload: Optional[dict[str, Any]] = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                payload = run_codex_batch(
                    args=args,
                    prompt=prompt,
                    schema_path=schema_path,
                    result_path=result_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    workspace_dir=batch_workspace_dir,
                )
                break
            except Exception as exc:
                last_error = exc
                log(f"batch {batch_id} size={len(batch)} codex attempt={attempt} failed: {exc}")
                time.sleep(min(2.0 * attempt, 10.0))

        if payload is None:
            raise RuntimeError(
                f"Codex sidecar generation failed for batch={batch_id}: {last_error}"
            ) from last_error

        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(f"Codex result for batch={batch_id} has no `results` list: {payload}")

        by_index: dict[int, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict) or "index" not in item:
                raise RuntimeError(f"Codex batch={batch_id} returned malformed item: {item!r}")
            by_index[int(item["index"])] = item

        expected_indices = {int(row["index"]) for row in batch}
        returned_indices = set(by_index.keys())
        if returned_indices != expected_indices:
            raise RuntimeError(
                f"Codex batch={batch_id} returned indices {sorted(returned_indices)} "
                f"but expected {sorted(expected_indices)}"
            )

        row_lines: list[str] = []
        for row in batch:
            idx = int(row["index"])
            bundle = model_validate(CandidateBundle, by_index[idx])
            normalized = normalize_bundle(
                bundle=bundle,
                answer=row["answer"],
                num_alternates=args.num_alternates,
                default_source=default_source,
            )
            row_lines.append(
                json.dumps(
                    build_sidecar_row(index=idx, normalized=normalized, args=args),
                    ensure_ascii=False,
                )
                + "\n"
            )
        return batch_id, row_lines

    progress = tqdm(
        total=len(batch_starts),
        desc=f"{progress_desc}:c{args.concurrent}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        with output_path.open(mode, encoding="utf-8") as fout:
            if args.concurrent == 1 or len(batch_starts) <= 1:
                for batch_id, start in enumerate(batch_starts, start=1):
                    _, row_lines = run_one_batch(batch_id, start)
                    for line in row_lines:
                        fout.write(line)
                        written += 1
                    fout.flush()
                    progress.update(1)
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    if written % 25 == 0:
                        log(f"codex rows_written={written}")
            else:
                next_batch_to_write = 1
                completed_batches: dict[int, list[str]] = {}
                max_workers = min(args.concurrent, len(batch_starts))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch_id = {
                        executor.submit(run_one_batch, batch_id, start): batch_id
                        for batch_id, start in enumerate(batch_starts, start=1)
                    }
                    for future in as_completed(future_to_batch_id):
                        batch_id, row_lines = future.result()
                        completed_batches[batch_id] = row_lines
                        progress.update(1)
                        while next_batch_to_write in completed_batches:
                            lines = completed_batches.pop(next_batch_to_write)
                            for line in lines:
                                fout.write(line)
                                written += 1
                            fout.flush()
                            if args.sleep_seconds > 0:
                                time.sleep(args.sleep_seconds)
                            if written % 25 == 0:
                                log(f"codex rows_written={written}")
                            next_batch_to_write += 1
    finally:
        progress.close()

    return written, status_text


def summarize_sidecar(path: Path) -> dict[str, Any]:
    rows = []
    seen: set[int] = set()
    duplicates: list[int] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            idx = int(row["index"])
            if idx in seen:
                duplicates.append(idx)
            seen.add(idx)
            alternates = row.get("alternates")
            if not isinstance(alternates, list) or not alternates:
                bad_rows += 1
                continue
            for key in (
                "scores",
                "relation_scores",
                "shared_fact_scores",
                "candidate_sources",
            ):
                value = row.get(key)
                if not isinstance(value, list) or len(value) != len(alternates):
                    bad_rows += 1
                    break
    return {
        "rows": len(rows),
        "unique_indices": len(seen),
        "duplicate_indices": duplicates,
        "bad_rows": bad_rows,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args)
    candidate_bank = load_candidate_bank(args.candidate_bank)
    done_indices = load_existing_indices(output_path) if args.resume else set()
    log(f"backend=codex_cli rows={len(rows)} already_done={len(done_indices)} output={output_path}")

    # Persist request metadata before generation so interrupted runs remain resumable.
    write_run_meta(output_path, args, codex_status=None)

    written, codex_status = generate_codex_sidecar(
        args=args,
        rows=rows,
        candidate_bank=candidate_bank,
        output_path=output_path,
        done_indices=done_indices,
    )

    write_run_meta(output_path, args, codex_status=codex_status)

    summary = summarize_sidecar(output_path)
    summary_path = output_path.with_name(output_path.name + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    log(
        f"done rows_written={written} total_rows={summary['rows']} "
        f"unique_indices={summary['unique_indices']} bad_rows={summary['bad_rows']}"
    )


if __name__ == "__main__":
    main()
