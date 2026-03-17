import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools import retry_invalid_counterfactuals


class RetryInvalidCounterfactualsTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            question_key="question",
            answer_key="answer",
            reject_gold_substring=True,
            require_short_answer=True,
            max_overlap_ratio=0.85,
            max_alt_length_chars=128,
        )

    def test_choose_retry_alternate_reranks_multi_candidate_response(self) -> None:
        alternate, candidates, scores, pick_meta = (
            retry_invalid_counterfactuals.choose_retry_alternate(
                self._args(),
                {
                    "question": "When was the event?",
                    "answer": "1999",
                    "candidate_answers": [],
                },
                {
                    "alternates": ["1998", "2001"],
                    "scores": [0.1, 0.9],
                    "answer_type": "year",
                },
            )
        )
        self.assertEqual(alternate, "2001")
        self.assertEqual(candidates, ["1998", "2001"])
        self.assertEqual(scores, [0.1, 0.9])
        self.assertEqual(pick_meta["external_score"], 0.9)

    def test_main_persists_retry_metadata_for_multi_candidate_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            row = {
                "index": 1,
                "question": "When was the event?",
                "answer": "1999",
                "alternate": "",
                "candidate_answers": ["1998", "2001"],
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            class _FakeGenerator:
                def many_sync(self, rows):
                    return [
                        {
                            "alternates": ["1998", "2001"],
                            "scores": [0.1, 0.9],
                            "same_relation": True,
                            "answer_type": "year",
                        }
                    ]

            argv = [
                "retry_invalid_counterfactuals.py",
                "--input-path",
                str(input_path),
                "--output-path",
                str(output_path),
                "--vllm-base-url",
                "http://localhost:8000/v1",
                "--vllm-model",
                "dummy-model",
                "--retry-passes",
                "1",
                "--num-alternates",
                "2",
                "--prompt-family",
                "strict_short",
                "--reject-gold-substring",
                "--require-short-answer",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    retry_invalid_counterfactuals,
                    "build_generator",
                    return_value=_FakeGenerator(),
                ):
                    retry_invalid_counterfactuals.main()

            saved_row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved_row["alternate"], "2001")
            self.assertEqual(saved_row["cf_retry_last_alternates"], ["1998", "2001"])
            self.assertEqual(saved_row["cf_retry_last_scores"], [0.1, 0.9])
            self.assertTrue(saved_row["cf_retry_success"])


if __name__ == "__main__":
    unittest.main()
