import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    build_answer_type_fallback_candidates,
    build_low_confidence_fallback_candidates,
    dedupe_scored_candidates,
)
from tools.make_counterfactuals import select_best_alternate


class DualCFV3CounterfactualSelectionTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "repair_invalid": True,
            "reject_gold_substring": True,
            "max_overlap_ratio": 0.85,
            "require_short_answer": True,
            "max_alt_length_chars": 128,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_dedupe_scored_candidates_keeps_best_numeric_score(self) -> None:
        candidates, scores = dedupe_scored_candidates(
            ["Rome", "Rome", "Milan"],
            [0.1, 0.9, None],
        )
        self.assertEqual(candidates, ["Rome", "Milan"])
        self.assertEqual(scores, [0.9, None])

    def test_select_best_alternate_uses_external_baseline_when_primary_pool_empty(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(repair_invalid=False, reject_gold_substring=False),
            question="What is the capital of Italy?",
            answer="Rome",
            seed=7,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["Milan", "Turin"],
            external_scores=[0.2, 0.9],
        )
        self.assertEqual(best_alt, "Turin")
        self.assertIsNone(invalid_reason)
        self.assertTrue(repaired)
        self.assertEqual(best_meta["external_score"], 0.9)

    def test_select_best_alternate_dedupes_external_candidates_before_ranking(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(repair_invalid=False, reject_gold_substring=False),
            question="What is the capital of Italy?",
            answer="Rome",
            seed=11,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["Milan", "Milan"],
            external_scores=[0.1, 0.9],
        )
        self.assertEqual(best_alt, "Milan")
        self.assertIsNone(invalid_reason)
        self.assertFalse(repaired)
        self.assertEqual(best_meta["external_score"], 0.9)

    def test_low_confidence_not_fallback_is_separate_from_primary_fallbacks(self) -> None:
        self.assertEqual(build_answer_type_fallback_candidates("Paris", seed=0), [])
        self.assertEqual(
            build_low_confidence_fallback_candidates("Paris"),
            ["not Paris"],
        )


if __name__ == "__main__":
    unittest.main()
