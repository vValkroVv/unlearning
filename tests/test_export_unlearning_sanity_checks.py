import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_eval_payload(prefix: str) -> dict:
    return {
        "forget_qa_rouge": {
            "agg_value": 0.5,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 0.1,
                    "input": f"system\n\nFacts.user\n\n{prefix} forget q0?assistant\n\n",
                    "ground_truth": f"{prefix} forget a0",
                    "generation": f"{prefix} forget gen0",
                },
                "1": {
                    "rougeL_recall": 0.2,
                    "input": f"system\n\nFacts.user\n\n{prefix} forget q1?assistant\n\n",
                    "ground_truth": f"{prefix} forget a1",
                    "generation": f"{prefix} forget gen1",
                },
                "2": {
                    "rougeL_recall": 0.3,
                    "input": f"system\n\nFacts.user\n\n{prefix} forget q2?assistant\n\n",
                    "ground_truth": f"{prefix} forget a2",
                    "generation": f"{prefix} forget gen2",
                },
            },
        },
        "holdout_qa_rouge": {
            "agg_value": 0.9,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 0.7,
                    "input": f"system\n\nFacts.user\n\n{prefix} holdout q0?assistant\n\n",
                    "ground_truth": f"{prefix} holdout a0",
                    "generation": f"{prefix} holdout gen0",
                },
                "1": {
                    "rougeL_recall": 0.8,
                    "input": f"system\n\nFacts.user\n\n{prefix} holdout q1?assistant\n\n",
                    "ground_truth": f"{prefix} holdout a1",
                    "generation": f"{prefix} holdout gen1",
                },
                "2": {
                    "rougeL_recall": 0.9,
                    "input": f"system\n\nFacts.user\n\n{prefix} holdout q2?assistant\n\n",
                    "ground_truth": f"{prefix} holdout a2",
                    "generation": f"{prefix} holdout gen2",
                },
            },
        },
    }


def build_cos_payload(prefix: str) -> dict:
    return {
        "forget_qa_cos_sim": {
            "agg_value": 0.4,
            "value_by_index": {
                "0": {"cos_sim": 0.11, "ground_truth": f"{prefix} forget a0", "generation": f"{prefix} forget gen0"},
                "1": {"cos_sim": 0.22, "ground_truth": f"{prefix} forget a1", "generation": f"{prefix} forget gen1"},
                "2": {"cos_sim": 0.33, "ground_truth": f"{prefix} forget a2", "generation": f"{prefix} forget gen2"},
            },
        },
        "holdout_qa_cos_sim": {
            "agg_value": 0.8,
            "value_by_index": {
                "0": {"cos_sim": 0.77, "ground_truth": f"{prefix} holdout a0", "generation": f"{prefix} holdout gen0"},
                "1": {"cos_sim": 0.88, "ground_truth": f"{prefix} holdout a1", "generation": f"{prefix} holdout gen1"},
                "2": {"cos_sim": 0.99, "ground_truth": f"{prefix} holdout a2", "generation": f"{prefix} holdout gen2"},
            },
        },
    }


class ExportUnlearningSanityChecksTest(unittest.TestCase):
    def _write_run(self, root: Path, algo_dir: str, run_name: str, prefix: str) -> None:
        eval_dir = root / "unlearn" / "duet" / algo_dir / run_name / "evals"
        hydra_dir = eval_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "DUET_EVAL.json").write_text(
            json.dumps(build_eval_payload(prefix)), encoding="utf-8"
        )
        (eval_dir / "COS_SIM_EVAL.json").write_text(
            json.dumps(build_cos_payload(prefix)), encoding="utf-8"
        )
        (eval_dir / "DUET_SUMMARY.json").write_text(
            json.dumps({"forget_qa_rouge": 0.5, "holdout_qa_rouge": 0.9}),
            encoding="utf-8",
        )
        (hydra_dir / "config.yaml").write_text(
            textwrap.dedent(
                """
                forget_split: city_forget_rare_5+city_forget_popular_5
                holdout_split: city_fast_retain_500
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    def test_exports_text_report_and_missing_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "saves-old"
            output_root = tmp / "reports"

            self._write_run(
                input_root,
                "ga",
                "duet_Llama-3.1-8B-Instruct_city_forget_5_ga_lora_r32_lalpha64_ldrop0p0_lr1e-4",
                "ga",
            )
            self._write_run(
                input_root,
                "npo",
                "duet_Llama-3.1-8B-Instruct_city_forget_5_npo_lora_r32_lalpha64_ldrop0p0_lr1e-4_beta0p5_alpha1p0_gamma1p0",
                "npo",
            )

            # Summary-only eval should be recorded as missing.
            missing_eval_dir = (
                input_root
                / "unlearn"
                / "duet"
                / "loku"
                / "duet_Llama-3.1-8B-Instruct_city_forget_5_loku_lora_r32_lalpha64_ldrop0p0_lr1e-4"
                / "evals"
            )
            missing_eval_dir.mkdir(parents=True, exist_ok=True)
            (missing_eval_dir / "DUET_SUMMARY.json").write_text(
                json.dumps({"forget_qa_rouge": 0.1}), encoding="utf-8"
            )

            subprocess.run(
                [
                    "python",
                    "src/tools/export_unlearning_sanity_checks.py",
                    "--input-root",
                    str(input_root),
                    "--lr",
                    "1e-4",
                    "--sample-count",
                    "2",
                    "--output-root",
                    str(output_root),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            report_index = (output_root / "report_index.tsv").read_text(encoding="utf-8")
            self.assertIn("city_forget_rare_5+city_forget_popular_5", report_index)
            report_path = next(output_root.glob("*.txt"))
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Benchmark: DUET", report_text)
            self.assertIn("GA", report_text)
            self.assertIn("NPO", report_text)
            self.assertIn("Answer to forget:", report_text)
            self.assertIn("rougeL_recall=", report_text)
            self.assertIn("cos_sim=", report_text)
            self.assertIn("Question: ga forget q", report_text)
            self.assertIn("Save path: ", report_text)
            self.assertIn(
                "saves-old/unlearn/duet/ga/duet_Llama-3.1-8B-Instruct_city_forget_5_ga_lora_r32_lalpha64_ldrop0p0_lr1e-4",
                report_text,
            )

            missing_table = (output_root / "missing_sample_logs.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("Found DUET_SUMMARY.json but no DUET_EVAL.json", missing_table)


if __name__ == "__main__":
    unittest.main()
