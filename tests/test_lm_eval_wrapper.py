import sys
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf


REPO_ROOT = Path("/workspace/unlearning")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evals.lm_eval import LMEvalEvaluator


class _FakeTaskManager:
    def __init__(self, include_path=None):
        self.include_path = include_path
        self.all_groups = set()


class _FakePeftModel:
    def __init__(self):
        self.peft_config = {"default": object()}
        self.eval_called = False
        self.tie_weights_calls = 0

    def eval(self):
        self.eval_called = True

    def tie_weights(self):
        self.tie_weights_calls += 1
        raise KeyError("attribute 'weight' already exists")


class _FakeHFLM:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        model.tie_weights()


class LMEvalWrapperTest(unittest.TestCase):
    def test_prepare_model_skips_second_tie_weights_for_peft_models(self) -> None:
        eval_cfg = OmegaConf.create(
            {
                "tasks": [],
                "overwrite": True,
                "include_subtask_metrics": False,
                "include_path": None,
            }
        )
        model = _FakePeftModel()

        with mock.patch("evals.lm_eval.TaskManager", _FakeTaskManager):
            evaluator = LMEvalEvaluator(eval_cfg)

        with mock.patch("evals.lm_eval.HFLM", _FakeHFLM):
            wrapped = evaluator.prepare_model(model, tokenizer="tok")

        self.assertTrue(model.eval_called)
        self.assertEqual(wrapped.kwargs["tokenizer"], "tok")
        self.assertEqual(model.tie_weights_calls, 0)

        with self.assertRaises(KeyError):
            model.tie_weights()
        self.assertEqual(model.tie_weights_calls, 1)


if __name__ == "__main__":
    unittest.main()
