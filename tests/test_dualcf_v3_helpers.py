import unittest
from pathlib import Path
import sys

import torch

TRAINER_ROOT = Path(__file__).resolve().parents[1] / "src" / "trainer"
if str(TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINER_ROOT))

from utils import build_answer_minus_shared_mask, build_answer_only_mask


class DualCFV3HelperTest(unittest.TestCase):
    def test_answer_only_mask_marks_supervised_tokens(self) -> None:
        inputs = {
            "labels": torch.tensor(
                [
                    [-100, -100, 10, 11, 12],
                    [-100, -100, -100, 20, 21],
                ]
            )
        }
        mask = build_answer_only_mask(inputs)
        expected = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_answer_minus_shared_mask_keeps_only_nonshared_answer_tokens(self) -> None:
        original_inputs = {
            "labels": torch.tensor([[-100, -100, 10, 11, 12]])
        }
        alternate_inputs = {
            "labels": torch.tensor([[-100, -100, 11, 13, -100]])
        }
        mask = build_answer_minus_shared_mask(original_inputs, alternate_inputs)
        expected = torch.tensor([[False, True, False, True]])
        self.assertTrue(torch.equal(mask, expected))

    def test_answer_minus_shared_mask_falls_back_when_everything_is_shared(self) -> None:
        original_inputs = {
            "labels": torch.tensor([[-100, -100, 10, 11]])
        }
        alternate_inputs = {
            "labels": torch.tensor([[-100, -100, 10, 11]])
        }
        mask = build_answer_minus_shared_mask(original_inputs, alternate_inputs)
        expected = build_answer_only_mask(original_inputs)
        self.assertTrue(torch.equal(mask, expected))


if __name__ == "__main__":
    unittest.main()
