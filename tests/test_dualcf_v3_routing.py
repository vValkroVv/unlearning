import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainer.unlearn.dual_cf import DualCF


class DualCFV3RoutingTest(unittest.TestCase):
    def _subject(self, **overrides):
        subject = DualCF.__new__(DualCF)
        subject.tau_d = 0.6
        subject.tau_a = 0.6
        subject.temp_d = 0.15
        subject.temp_a = 0.15
        subject.neg_power = 1.0
        subject.risk_power = 1.0
        subject.disable_difficulty_route = False
        subject.disable_attribution_route = False
        for key, value in overrides.items():
            setattr(subject, key, value)
        return subject

    def _forget_inputs(self):
        return {
            "difficulty_score": torch.tensor([0.2, 0.9], dtype=torch.float32),
            "attribution_score": torch.tensor([0.8, 0.1], dtype=torch.float32),
        }

    def test_disable_difficulty_route_forces_unit_difficulty_gate(self) -> None:
        subject = self._subject(disable_difficulty_route=True)
        _, _, difficulty_gate, _ = subject._routing_state(
            self._forget_inputs(),
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertTrue(torch.equal(difficulty_gate, torch.ones_like(difficulty_gate)))

    def test_disable_attribution_route_forces_zero_risk_gate(self) -> None:
        subject = self._subject(disable_attribution_route=True)
        _, _, _, risk_gate = subject._routing_state(
            self._forget_inputs(),
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertTrue(torch.equal(risk_gate, torch.zeros_like(risk_gate)))

    def test_disabled_routes_keep_raw_scores_for_logging(self) -> None:
        subject = self._subject(
            disable_difficulty_route=True,
            disable_attribution_route=True,
        )
        difficulty, attribution, difficulty_gate, risk_gate = subject._routing_state(
            self._forget_inputs(),
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertTrue(
            torch.equal(difficulty, torch.tensor([0.2, 0.9], dtype=torch.float32))
        )
        self.assertTrue(
            torch.equal(attribution, torch.tensor([0.8, 0.1], dtype=torch.float32))
        )
        self.assertTrue(torch.equal(difficulty_gate, torch.ones_like(difficulty_gate)))
        self.assertTrue(torch.equal(risk_gate, torch.zeros_like(risk_gate)))


if __name__ == "__main__":
    unittest.main()
