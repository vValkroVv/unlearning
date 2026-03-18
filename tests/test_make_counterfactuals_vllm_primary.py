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

from tools import make_counterfactuals


class _FakeVLLMGenerator:
    def __init__(self, responses):
        self._responses = list(responses)
        self._cursor = 0

    def many_sync(self, rows):
        batch = self._responses[self._cursor : self._cursor + len(rows)]
        self._cursor += len(rows)
        return batch


class MakeCounterfactualsVLLMPrimaryTest(unittest.TestCase):
    def test_vllm_primary_keeps_per_row_candidate_bank_metadata(self) -> None:
        dataset_rows = [
            {
                "index": 1,
                "question": "When was event alpha?",
                "answer": "1999",
            },
            {
                "index": 2,
                "question": "When was event beta?",
                "answer": "1999",
            },
        ]
        candidate_bank = {
            "1": {
                "candidate_answers": ["1998", "2001"],
                "candidate_relation_scores": [1.0, 0.0],
                "candidate_shared_fact_scores": [1.0, 0.0],
                "candidate_sources": ["bank_row1", "bank_row1"],
            },
            "2": {
                "candidate_answers": ["1998", "2001"],
                "candidate_relation_scores": [0.0, 1.0],
                "candidate_shared_fact_scores": [0.0, 1.0],
                "candidate_sources": ["bank_row2", "bank_row2"],
            },
        }
        responses = [
            {
                "alternates": [],
                "same_relation": True,
                "answer_type": "year",
            },
            {
                "alternates": [],
                "same_relation": True,
                "answer_type": "year",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "artifact.jsonl"
            argv = [
                "make_counterfactuals.py",
                "--dataset-path",
                "ignored",
                "--split",
                "train",
                "--output-path",
                str(output_path),
                "--generator-backend",
                "vllm_openai",
                "--vllm-base-url",
                "http://localhost:8000/v1",
                "--vllm-model",
                "dummy-model",
                "--candidate-bank",
                "ignored.jsonl",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    make_counterfactuals,
                    "load_dataset_split",
                    return_value=dataset_rows,
                ):
                    with mock.patch.object(
                        make_counterfactuals,
                        "maybe_load_candidate_bank",
                        return_value=candidate_bank,
                    ):
                        with mock.patch.object(
                            make_counterfactuals,
                            "build_vllm_generator",
                            return_value=_FakeVLLMGenerator(responses),
                        ):
                            make_counterfactuals.main()

            saved_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(saved_rows), 2)
        self.assertEqual(saved_rows[0]["alternate"], "1998")
        self.assertEqual(saved_rows[1]["alternate"], "2001")
        self.assertEqual(saved_rows[0]["candidate_relation_scores"], [1.0, 0.0])
        self.assertEqual(saved_rows[1]["candidate_relation_scores"], [0.0, 1.0])
        self.assertEqual(saved_rows[0]["candidate_shared_fact_scores"], [1.0, 0.0])
        self.assertEqual(saved_rows[1]["candidate_shared_fact_scores"], [0.0, 1.0])
        self.assertEqual(saved_rows[0]["candidate_sources"], ["bank_row1", "bank_row1"])
        self.assertEqual(saved_rows[1]["candidate_sources"], ["bank_row2", "bank_row2"])
        self.assertEqual(saved_rows[0]["cf_pick_meta"]["selected_source"], "bank_row1")
        self.assertEqual(saved_rows[1]["cf_pick_meta"]["selected_source"], "bank_row2")


if __name__ == "__main__":
    unittest.main()
