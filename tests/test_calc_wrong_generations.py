import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_eval_payload() -> dict:
    return {
        "forget_qa_rouge": {
            "agg_value": 0.15,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 0.0,
                    "input": "system\n\nFacts.user\n\nforget q0?assistant\n\n",
                    "ground_truth": "forget a0",
                    "generation": "",
                },
                "1": {
                    "rougeL_recall": 1.0,
                    "input": "system\n\nFacts.user\n\nforget q1?assistant\n\n",
                    "ground_truth": "forget a1",
                    "generation": "forget a1",
                },
            },
        },
        "holdout_qa_rouge": {
            "agg_value": 0.85,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 1.0,
                    "input": "system\n\nFacts.user\n\nholdout q0?assistant\n\n",
                    "ground_truth": "holdout a0",
                    "generation": "holdout a0",
                }
            },
        },
    }


class CalcWrongGenerationsTest(unittest.TestCase):
    def test_writes_eval_and_summary_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = (
                tmp
                / "saves"
                / "unlearn"
                / "duet_Llama-3.1-8B-Instruct_city_forget_rare_5_span_cf_samnpo_lora_lr1e-4_seed42"
            )
            final_eval_dir = run_dir / "evals"
            checkpoint_eval_dir = run_dir / "checkpoint_evals" / "checkpoint-10"
            final_eval_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_eval_dir.mkdir(parents=True, exist_ok=True)

            payload = build_eval_payload()
            (final_eval_dir / "DUET_EVAL.json").write_text(json.dumps(payload), encoding="utf-8")
            (checkpoint_eval_dir / "DUET_EVAL.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            subprocess.run(
                [
                    "python",
                    "scripts/calc_wrong_generations.py",
                    "--path_to_saves",
                    str(tmp / "saves"),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            for eval_dir in (final_eval_dir, checkpoint_eval_dir):
                detail_payload = json.loads(
                    (eval_dir / "WRONG_GENERATIONS_EVAL.json").read_text(encoding="utf-8")
                )
                summary_payload = json.loads(
                    (eval_dir / "WRONG_GENERATIONS_SUMMARY.json").read_text(encoding="utf-8")
                )

                self.assertAlmostEqual(summary_payload["forget_wrong_gen_rate"], 0.5)
                self.assertAlmostEqual(summary_payload["holdout_wrong_gen_rate"], 0.0)

                forget_detail = detail_payload["forget_wrong_gen_rate"]
                self.assertAlmostEqual(forget_detail["agg_value"], 0.5)
                self.assertEqual(forget_detail["wrong_count"], 1)
                self.assertEqual(forget_detail["total_count"], 2)
                self.assertEqual(forget_detail["reason_counts"]["empty_like"], 1)
                self.assertTrue(forget_detail["value_by_index"]["0"]["wrong_generation"])
                self.assertEqual(
                    forget_detail["value_by_index"]["0"]["question"],
                    "forget q0?",
                )
                self.assertFalse(forget_detail["value_by_index"]["1"]["wrong_generation"])


if __name__ == "__main__":
    unittest.main()
