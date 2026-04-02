import csv
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "src" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_results_combine_tables import load_table_bundle


def build_eval_payload() -> dict:
    return {
        "forget_qa_rouge": {
            "agg_value": 0.25,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 0.0,
                    "input": "system\n\nFacts.user\n\nforget q0?assistant\n\n",
                    "ground_truth": "forget answer 0",
                    "generation": "",
                },
                "1": {
                    "rougeL_recall": 0.5,
                    "input": "system\n\nFacts.user\n\nforget q1?assistant\n\n",
                    "ground_truth": "forget answer 1",
                    "generation": "forget answer 1",
                },
            },
        },
        "holdout_qa_rouge": {
            "agg_value": 0.75,
            "value_by_index": {
                "0": {
                    "rougeL_recall": 1.0,
                    "input": "system\n\nFacts.user\n\nholdout q0?assistant\n\n",
                    "ground_truth": "holdout answer 0",
                    "generation": "holdout answer 0",
                },
                "1": {
                    "rougeL_recall": 0.0,
                    "input": "system\n\nFacts.user\n\nholdout q1?assistant\n\n",
                    "ground_truth": "holdout answer 1",
                    "generation": "assistant\n\nholdout leak",
                },
            },
        },
    }


def write_run_tree(root: Path) -> Path:
    run_name = (
        "duet_Llama-3.1-8B-Instruct_city_forget_rare_5_"
        "span_cf_samnpo_lora_r32_lalpha64_ldrop0p0_lr1e-4_seed42"
    )
    run_dir = root / "unlearn" / run_name
    eval_dir = run_dir / "evals"
    checkpoint_eval_dir = run_dir / "checkpoint_evals" / "checkpoint-100"
    hydra_dir = run_dir / ".hydra"

    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)

    payload = build_eval_payload()
    summary = {"forget_qa_rouge": 0.25, "holdout_qa_rouge": 0.75}
    for target_dir in (eval_dir, checkpoint_eval_dir):
        (target_dir / "DUET_EVAL.json").write_text(json.dumps(payload), encoding="utf-8")
        (target_dir / "DUET_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")

    (hydra_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            forget_split: city_forget_rare_5
            holdout_split: city_fast_retain_500
            trainer:
              args:
                num_train_epochs: 5
              save_on_epochs:
                - 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return run_dir


class WrongGenerationSidecarsTest(unittest.TestCase):
    def test_sidecar_script_feeds_structured_saves_and_bundle_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            saves_root = tmp / "saves"
            run_dir = write_run_tree(saves_root)

            subprocess.run(
                [
                    "python",
                    "scripts/calc_wrong_generations.py",
                    "--path_to_saves",
                    str(saves_root),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            for eval_dir in (run_dir / "evals", run_dir / "checkpoint_evals" / "checkpoint-100"):
                summary_path = eval_dir / "WRONG_GENERATIONS_SUMMARY.json"
                eval_path = eval_dir / "WRONG_GENERATIONS_EVAL.json"
                self.assertTrue(summary_path.exists())
                self.assertTrue(eval_path.exists())

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertAlmostEqual(summary["forget_wrong_gen_rate"], 0.5)
                self.assertAlmostEqual(summary["holdout_wrong_gen_rate"], 0.5)

            structured_root = tmp / "structured-saves"
            subprocess.run(
                [
                    "python",
                    "src/tools/build_structured_saves.py",
                    "--input-root",
                    str(saves_root / "unlearn"),
                    "--output-root",
                    str(structured_root),
                    "--overwrite",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            wrong_rate_tsv = structured_root / "duet_rare" / "1e-4" / "forget_wrong_gen_rate.tsv"
            self.assertTrue(wrong_rate_tsv.exists())

            with wrong_rate_tsv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "span_cf_samnpo")
            self.assertIn("0.5", set(rows[0].values()))

            bundle = load_table_bundle(
                structured_root,
                split="duet_rare",
                lr="1e-4",
                metrics=[
                    ("forget_qa_rouge", "F"),
                    ("holdout_qa_rouge", "H"),
                    ("forget_wrong_gen_rate", "FW"),
                    ("holdout_wrong_gen_rate", "HW"),
                ],
                wrong_generation_index={},
                wrong_generation_label=None,
            )

            self.assertEqual(
                bundle["forget_wrong_gen_rate"]["span_cf_samnpo"]["5.0"],
                "0.5",
            )
            self.assertEqual(
                bundle["holdout_wrong_gen_rate"]["span_cf_samnpo"]["5.0"],
                "0.5",
            )


if __name__ == "__main__":
    unittest.main()
