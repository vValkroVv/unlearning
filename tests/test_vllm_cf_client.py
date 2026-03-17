import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.vllm_cf_client import VLLMCFGenerator


class VLLMCFClientTest(unittest.TestCase):
    def _generator(self, **overrides):
        values = {
            "base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "model": "dummy-model",
            "prompt_family": "strict_short",
            "num_alternates": 4,
        }
        values.update(overrides)
        return VLLMCFGenerator(**values)

    def test_parse_plain_text_response_normalizes_single_alternate(self) -> None:
        generator = self._generator()
        payload = generator._normalize_response(
            alternates=["Milan"],
            same_relation=True,
            answer_type="plain_text",
        )
        self.assertEqual(payload["alternate"], "Milan")
        self.assertEqual(payload["alternates"], ["Milan"])
        self.assertEqual(payload["scores"], [])

    def test_parse_single_json_response_is_backward_compatible(self) -> None:
        generator = self._generator()
        payload = generator._parse_payload(
            '{"alternate":"Milan","same_relation":true,"answer_type":"city"}'
        )
        self.assertEqual(payload["alternate"], "Milan")
        self.assertEqual(payload["alternates"], ["Milan"])
        self.assertEqual(payload["answer_type"], "city")

    def test_parse_multi_json_response_preserves_alternates_and_scores(self) -> None:
        generator = self._generator()
        payload = generator._parse_payload(
            '{"alternates":["Milan","Turin"],"scores":[0.2,0.9],"same_relation":true,"answer_type":"city"}'
        )
        self.assertEqual(payload["alternate"], "Milan")
        self.assertEqual(payload["alternates"], ["Milan", "Turin"])
        self.assertEqual(payload["scores"], [0.2, 0.9])

    def test_parse_multi_json_response_normalizes_malformed_scores(self) -> None:
        generator = self._generator()
        payload = generator._parse_payload(
            '{"alternates":["Milan","Turin"],"scores":["bad"],"same_relation":true,"answer_type":"city"}'
        )
        self.assertEqual(payload["scores"], [None, None])

    def test_build_messages_reflects_prompt_family_and_count(self) -> None:
        with mock.patch.dict(os.environ, {"VLLM_USE_STRUCTURED_OUTPUTS": "1"}, clear=False):
            generator = self._generator(prompt_family="rwku_shared_fact_safe", num_alternates=3)
            messages = generator.build_messages(
                question="When was the event?",
                answer="1999",
            )
        self.assertIn("shared facts", messages[0]["content"])
        self.assertIn("Generate up to 3", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
