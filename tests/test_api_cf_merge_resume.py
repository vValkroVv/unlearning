import argparse
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_codex_phase_a_resume():
    module_path = REPO_ROOT / "scripts" / "api_cf" / "codex_phase_a_resume.py"
    spec = importlib.util.spec_from_file_location("codex_phase_a_resume", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ApiCFMergeResumeTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_merge_allows_mixed_backends_and_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            left_dir = tmp / "left"
            right_dir = tmp / "right"
            left_dir.mkdir()
            right_dir.mkdir()

            left_rows = [
                {
                    "index": 0,
                    "alternates": ["Paris", "Madrid"],
                    "scores": [0.9, 0.8],
                    "relation_scores": [1.0, 1.0],
                    "shared_fact_scores": [1.0, 1.0],
                    "candidate_sources": ["left:a", "left:b"],
                    "generator_backend": "codex_cli",
                    "generator_model": "gpt-5.4-mini",
                    "generator_reasoning_effort": "medium",
                    "model": "gpt-5.4-mini",
                    "prompt_family": "duet_relation_safe",
                }
            ]
            right_rows = [
                {
                    "index": 0,
                    "alternates": ["Rome", "Lisbon"],
                    "scores": [0.7, 0.6],
                    "relation_scores": [1.0, 1.0],
                    "shared_fact_scores": [1.0, 1.0],
                    "candidate_sources": ["right:a", "right:b"],
                    "generator_backend": "chatgpt_manual",
                    "generator_model": "gpt-5.4-pro",
                    "generator_reasoning_effort": "xhigh",
                    "model": "gpt-5.4-pro",
                    "prompt_family": "duet_relation_safe",
                }
            ]
            left_meta = {
                "backend": "codex_cli",
                "model": "gpt-5.4-mini",
                "reasoning_effort": "medium",
                "dataset_path": "SwetieePawsss/exp_r",
                "dataset_name": "forget_level2",
                "split": "test",
                "data_files": None,
                "question_key": "query",
                "answer_key": "answer",
                "answer_index": None,
                "prompt_family": "duet_relation_safe",
                "num_alternates": 2,
            }
            right_meta = {
                "backend": "chatgpt_manual",
                "model": "gpt-5.4-pro",
                "reasoning_effort": "xhigh",
                "dataset_path": "/mnt/data/RWKU_unzipped/RWKU",
                "dataset_name": "forget_level2",
                "split": "test",
                "data_files": None,
                "question_key": "query",
                "answer_key": "answer",
                "answer_index": None,
                "prompt_family": "duet_relation_safe",
                "num_alternates": 2,
            }
            self._write_jsonl(left_dir / "api_sidecar.jsonl", left_rows)
            self._write_jsonl(right_dir / "api_sidecar.jsonl", right_rows)
            (left_dir / "api_sidecar.jsonl.meta.json").write_text(
                json.dumps(left_meta), encoding="utf-8"
            )
            (right_dir / "api_sidecar.jsonl.meta.json").write_text(
                json.dumps(right_meta), encoding="utf-8"
            )

            out_path = tmp / "merged" / "api_sidecar.jsonl"
            subprocess.run(
                [
                    "python",
                    "scripts/api_cf/merge_codex_sidecars.py",
                    "--input-dir",
                    str(left_dir),
                    "--input-dir",
                    str(right_dir),
                    "--output-path",
                    str(out_path),
                    "--max-alternates",
                    "4",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            merged_meta = json.loads(
                out_path.with_name("api_sidecar.jsonl.meta.json").read_text(encoding="utf-8")
            )
            merged_row = json.loads(out_path.read_text(encoding="utf-8").strip())
            self.assertEqual(merged_meta["model"], "multiple")
            self.assertEqual(merged_meta["backend"], "multiple")
            self.assertEqual(merged_meta["dataset_path"], "SwetieePawsss/exp_r")
            self.assertEqual(
                merged_meta["input_backends"], ["codex_cli", "chatgpt_manual"]
            )
            self.assertEqual(
                merged_meta["input_dataset_paths"],
                ["SwetieePawsss/exp_r", "/mnt/data/RWKU_unzipped/RWKU"],
            )
            self.assertEqual(merged_row["generator_backend"], "multiple")
            self.assertEqual(
                merged_row["merged_backends"], ["codex_cli", "chatgpt_manual"]
            )

    def test_resume_validate_accepts_merged_sidecar_model_and_dataset_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            meta_path = tmp / "api_sidecar.jsonl.meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "merged_from_sidecars": True,
                        "model": "multiple",
                        "input_models": ["gpt-5.4-mini", "gpt-5.4-pro"],
                        "prompt_family": "rwku_shared_fact_safe",
                        "dataset_path": "/mnt/data/RWKU_unzipped/RWKU",
                        "input_dataset_paths": [
                            "SwetieePawsss/exp_r",
                            "/mnt/data/RWKU_unzipped/RWKU",
                        ],
                        "dataset_name": "forget_level2",
                        "split": "test",
                        "data_files": None,
                        "question_key": "query",
                        "answer_key": "answer",
                        "answer_index": None,
                        "num_alternates": 8,
                    }
                ),
                encoding="utf-8",
            )

            module = load_codex_phase_a_resume()
            args = argparse.Namespace(
                model="gpt-5.4-mini",
                prompt_family="rwku_shared_fact_safe",
                dataset_path="SwetieePawsss/exp_r",
                dataset_name="forget_level2",
                split="test",
                data_files=None,
                question_key="query",
                answer_key="answer",
                answer_index=None,
                num_alternates=8,
            )

            module.validate_meta(args, meta_path)


if __name__ == "__main__":
    unittest.main()
