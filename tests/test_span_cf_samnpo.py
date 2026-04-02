import importlib.machinery
import sys
import types
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "deepspeed" not in sys.modules:
    deepspeed_stub = types.ModuleType("deepspeed")
    deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="deepspeed",
        loader=None,
    )
    sys.modules["deepspeed"] = deepspeed_stub

from trainer.unlearn.span_cf_samnpo import SpanCFSAMNPO


class SpanCFSAMNPOBranchScaleTest(unittest.TestCase):
    def _make_trainer_shell(
        self,
        cf_branch_scale: float,
        neg_branch_scale: float,
    ) -> SpanCFSAMNPO:
        trainer = SpanCFSAMNPO.__new__(SpanCFSAMNPO)
        trainer.cf_branch_scale = cf_branch_scale
        trainer.neg_branch_scale = neg_branch_scale
        return trainer

    def test_apply_branch_scales_reweights_cf_and_neg_losses(self) -> None:
        trainer = self._make_trainer_shell(cf_branch_scale=0.8, neg_branch_scale=1.2)

        cf_scaled, neg_scaled = trainer._apply_branch_scales(
            cf_loss=torch.tensor(10.0),
            neg_loss=torch.tensor(5.0),
        )

        self.assertAlmostEqual(float(cf_scaled.item()), 8.0)
        self.assertAlmostEqual(float(neg_scaled.item()), 6.0)

    def test_apply_branch_scales_preserves_defaults(self) -> None:
        trainer = self._make_trainer_shell(cf_branch_scale=1.0, neg_branch_scale=1.0)

        cf_scaled, neg_scaled = trainer._apply_branch_scales(
            cf_loss=torch.tensor(3.25),
            neg_loss=torch.tensor(7.5),
        )

        self.assertAlmostEqual(float(cf_scaled.item()), 3.25)
        self.assertAlmostEqual(float(neg_scaled.item()), 7.5)

    def test_duet_dual_cf_dataset_config_requests_required_routing_keys(self) -> None:
        config_path = REPO_ROOT / "configs/data/datasets/DUET_QA_forget_dual_cf.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        dataset_args = config["DUET_QA_forget_dual_cf"]["args"]

        self.assertEqual(dataset_args["alternate_key"], "alternate")
        self.assertEqual(
            dataset_args["metadata_keys"],
            ["difficulty_score", "attribution_score", "index"],
        )
        self.assertIn("rarity_score", dataset_args["optional_metadata_keys"])


if __name__ == "__main__":
    unittest.main()
