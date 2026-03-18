import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools import validate_dual_cf_artifact


class ValidateDualCFArtifactTest(unittest.TestCase):
    def _write_rows(self, path: Path, rows) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_report_counts_duplicate_raw_candidates_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.jsonl"
            report_path = Path(tmpdir) / "report.json"
            self._write_rows(
                artifact_path,
                [
                    {
                        "index": 1,
                        "question": "When was the event?",
                        "answer": "1999",
                        "alternate": "2001",
                        "difficulty_score": 0.2,
                        "attribution_score": 0.3,
                        "candidate_answers": [],
                        "external_alternates": ["1998", "1998", "2001"],
                        "external_alternate_scores": [0.1, 0.2, 0.9],
                        "external_alternate_relation_scores": [0.5, 0.6, 1.0],
                        "external_alternate_shared_fact_scores": [0.5, 0.6, 1.0],
                        "external_alternate_sources": ["sidecar", "sidecar", "sidecar"],
                        "cf_pick_meta": {
                            "selected_candidate": "2001",
                            "selected_candidate_index": 1,
                            "selected_source": "sidecar",
                            "selected_from_pool": "sidecar",
                            "candidate_pool_size": 2,
                            "duplicate_candidates_removed": 1,
                            "rank_score": 0.9,
                            "used_low_confidence_fallback": False,
                        },
                        "cf_invalid_reason": None,
                        "cf_is_valid": True,
                        "cf_provenance": {
                            "generator_backend": "vllm_openai",
                            "prompt_family": "duet_relation_safe",
                            "candidate_count": 4,
                            "prompt_version": "vllm_openai:duet_relation_safe:v1",
                        },
                    }
                ],
            )
            argv = [
                "validate_dual_cf_artifact.py",
                "--input-path",
                str(artifact_path),
                "--question-key",
                "question",
                "--strict",
                "--reject-gold-substring",
                "--require-short-answer",
                "--report-path",
                str(report_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                result = validate_dual_cf_artifact.main()

            self.assertEqual(result, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["artifact_quality"]["duplicate_external_candidate_total"],
                1,
            )
            self.assertEqual(
                report["artifact_quality"]["relation_metadata_coverage_rate"],
                1.0,
            )

    def test_selected_candidate_mismatch_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.jsonl"
            self._write_rows(
                artifact_path,
                [
                    {
                        "index": 1,
                        "question": "When was the event?",
                        "answer": "1999",
                        "alternate": "2001",
                        "difficulty_score": 0.2,
                        "attribution_score": 0.3,
                        "cf_pick_meta": {
                            "selected_candidate": "1998",
                            "selected_candidate_index": 0,
                            "selected_source": "sidecar",
                            "candidate_pool_size": 2,
                        },
                    }
                ],
            )
            argv = [
                "validate_dual_cf_artifact.py",
                "--input-path",
                str(artifact_path),
                "--question-key",
                "question",
                "--strict",
            ]
            with mock.patch.object(sys, "argv", argv):
                result = validate_dual_cf_artifact.main()

            self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
