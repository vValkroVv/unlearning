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
            "allow_low_confidence_fallback": False,
            "reject_gold_substring": True,
            "max_overlap_ratio": 0.85,
            "require_short_answer": True,
            "max_alt_length_chars": 128,
            "prompt_family": "default",
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
        self.assertEqual(best_meta["candidate_pool_size"], 2)

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
        self.assertEqual(best_meta["duplicate_candidates_removed"], 1)

    def test_relation_and_shared_fact_scores_can_break_external_score_ties(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(repair_invalid=False, reject_gold_substring=False),
            question="When was the event?",
            answer="1999",
            seed=3,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["1998", "2001"],
            external_scores=[0.5, 0.5],
            external_relation_scores=[0.1, 1.0],
            external_shared_fact_scores=[0.1, 1.0],
            external_sources=["sidecar", "sidecar"],
        )
        self.assertEqual(best_alt, "2001")
        self.assertIsNone(invalid_reason)
        self.assertTrue(repaired)
        self.assertEqual(best_meta["relation_score"], 1.0)
        self.assertEqual(best_meta["shared_fact_score"], 1.0)
        self.assertEqual(best_meta["selected_source"], "sidecar")

    def test_low_confidence_fallback_requires_explicit_flag(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(
                reject_gold_substring=False,
                allow_low_confidence_fallback=False,
            ),
            question="What is the capital of France?",
            answer="Paris",
            seed=0,
            primary_candidates=[],
            row_candidates=[],
        )
        self.assertEqual(best_alt, "")
        self.assertEqual(invalid_reason, "empty")
        self.assertFalse(repaired)
        self.assertEqual(best_meta["invalid_reason"], "no_candidates")

        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(
                reject_gold_substring=False,
                allow_low_confidence_fallback=True,
            ),
            question="What is the capital of France?",
            answer="Paris",
            seed=0,
            primary_candidates=[],
            row_candidates=[],
        )
        self.assertEqual(best_alt, "not Paris")
        self.assertIsNone(invalid_reason)
        self.assertTrue(repaired)
        self.assertTrue(best_meta["used_low_confidence_fallback"])

    def test_low_confidence_not_fallback_is_separate_from_primary_fallbacks(self) -> None:
        self.assertEqual(build_answer_type_fallback_candidates("Paris", seed=0), [])
        self.assertEqual(
            build_low_confidence_fallback_candidates("Paris"),
            ["not Paris"],
        )

    def test_rwku_relation_rescue_prefers_high_relation_candidate(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(
                repair_invalid=False,
                reject_gold_substring=False,
                prompt_family="rwku_shared_fact_safe",
            ),
            question="What year was Stephen King awarded the Medal for Distinguished Contribution to American Letters?",
            answer="2003",
            seed=17,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["2004", "1973"],
            external_scores=[0.95, 0.0],
            external_relation_scores=[0.05, 0.86],
            external_shared_fact_scores=[0.20, 0.0],
            external_sources=["filmography", "candidate_bank"],
        )
        self.assertEqual(best_alt, "1973")
        self.assertIsNone(invalid_reason)
        self.assertTrue(repaired)
        self.assertTrue(best_meta["rwku_relation_rescue_applied"])
        self.assertEqual(best_meta["selected_source"], "candidate_bank")

    def test_non_rwku_prompt_keeps_original_argmax(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(
                repair_invalid=False,
                reject_gold_substring=False,
                prompt_family="default",
            ),
            question="What year was Stephen King awarded the Medal for Distinguished Contribution to American Letters?",
            answer="2003",
            seed=17,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["2004", "1973"],
            external_scores=[0.95, 0.0],
            external_relation_scores=[0.05, 0.86],
            external_shared_fact_scores=[0.20, 0.0],
            external_sources=["forget_semantic_nn", "candidate_bank"],
        )
        self.assertEqual(best_alt, "2004")
        self.assertIsNone(invalid_reason)
        self.assertFalse(repaired)
        self.assertEqual(best_meta["selected_source"], "forget_semantic_nn")

    def test_rwku_relation_rescue_respects_gold_substring_guard(self) -> None:
        best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
            args=self._args(
                repair_invalid=False,
                reject_gold_substring=True,
                prompt_family="rwku_shared_fact_safe",
            ),
            question="Which NCAA Division did John Cena play football in at college?",
            answer="Division III",
            seed=5,
            primary_candidates=[],
            row_candidates=[],
            external_candidates=["Division I", "Division II", "University of Southern Mississippi"],
            external_scores=[0.92, 0.89, 0.67],
            external_relation_scores=[0.96, 0.94, 0.40],
            external_shared_fact_scores=[0.91, 0.90, 0.86],
            external_sources=[
                "college_football_divisions",
                "college_football_divisions",
                "forget_semantic_nn",
            ],
        )
        self.assertEqual(best_alt, "University of Southern Mississippi")
        self.assertIsNone(invalid_reason)
        self.assertTrue(repaired)
        self.assertEqual(best_meta["selected_source"], "forget_semantic_nn")
        self.assertNotIn("rwku_relation_rescue_applied", best_meta)


if __name__ == "__main__":
    unittest.main()
