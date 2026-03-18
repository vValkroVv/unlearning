import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)
LM_EVAL_AVAILABLE = importlib.util.find_spec("lm_eval") is not None


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_tokenizer(output_dir: Path) -> Path:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "A": 4,
        "B": 5,
        "C": 6,
        "D": 7,
        "Question": 8,
        "Answer": 9,
        "Choose": 10,
        "the": 11,
        "correct": 12,
        "option": 13,
        ".": 14,
        "only": 15,
        "sky": 16,
        "blue": 17,
        "green": 18,
        "two": 19,
        "four": 20,
        "science": 21,
        "planet": 22,
        "truth": 23,
        "commonsense": 24,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    return output_dir


def make_tiny_llama_model(model_dir: Path) -> Path:
    config = LlamaConfig(
        vocab_size=25,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir)
    return model_dir


def make_lora_adapter(base_model_dir: Path, output_dir: Path) -> Path:
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=8,
            lora_dropout=0.0,
            r=4,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    peft_model.save_pretrained(output_dir)
    return output_dir


class UtilityPipelineTest(unittest.TestCase):
    def test_build_utility_panel_with_local_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "panel"

            write_jsonl(
                tmp_path / "mmlu.jsonl",
                [
                    {
                        "question_id": 1,
                        "question": "math one",
                        "options": ["A", "B", "C", "D"],
                        "answer_index": 0,
                        "category": "math",
                    },
                    {
                        "question_id": 2,
                        "question": "math blocked person",
                        "options": ["A", "B", "C", "D"],
                        "answer_index": 1,
                        "category": "math",
                    },
                    {
                        "question_id": 3,
                        "question": "history one",
                        "options": ["A", "B", "C", "D"],
                        "answer_index": 2,
                        "category": "history",
                    },
                    {
                        "question_id": 4,
                        "question": "science one",
                        "options": ["A", "B", "C", "D"],
                        "answer_index": 3,
                        "category": "science",
                    },
                ],
            )
            write_jsonl(
                tmp_path / "truthful.jsonl",
                [
                    {"question": "truth one", "choices": ["yes", "no"], "label": 0},
                    {"question": "truth two", "choices": ["yes", "no"], "label": 1},
                    {
                        "question": "truth blocked person",
                        "choices": ["yes", "no"],
                        "label": 1,
                    },
                ],
            )
            write_jsonl(
                tmp_path / "arc.jsonl",
                [
                    {
                        "id": "arc1",
                        "question": "science one",
                        "choices": {"text": ["A", "B", "C", "D"], "label": ["A", "B", "C", "D"]},
                        "answerKey": "A",
                    },
                    {
                        "id": "arc2",
                        "question": "science two",
                        "choices": {"text": ["A", "B", "C", "D"], "label": ["A", "B", "C", "D"]},
                        "answerKey": "B",
                    },
                    {
                        "id": "arc3",
                        "question": "science blocked person",
                        "choices": {"text": ["A", "B", "C", "D"], "label": ["A", "B", "C", "D"]},
                        "answerKey": "C",
                    },
                ],
            )
            write_jsonl(
                tmp_path / "wino.jsonl",
                [
                    {
                        "id": "wino1",
                        "sentence": "commonsense one",
                        "option1": "A",
                        "option2": "B",
                        "answer": "1",
                    },
                    {
                        "id": "wino2",
                        "sentence": "commonsense two",
                        "option1": "A",
                        "option2": "B",
                        "answer": "2",
                    },
                    {
                        "id": "wino3",
                        "sentence": "commonsense blocked person",
                        "option1": "A",
                        "option2": "B",
                        "answer": "1",
                    },
                ],
            )
            alias_path = tmp_path / "aliases.txt"
            alias_path.write_text("blocked person\n", encoding="utf-8")

            cmd = [
                str(PYTHON_BIN),
                "src/tools/build_utility_1k_panel.py",
                "--output-dir",
                str(output_dir),
                "--seed",
                "7",
                "--mmlu-pro",
                "3",
                "--truthfulqa-bin",
                "2",
                "--arc",
                "2",
                "--winogrande",
                "2",
                "--exclude-targets-file",
                str(alias_path),
                "--mmlu-pro-path",
                "json",
                "--mmlu-pro-split",
                "train",
                "--mmlu-pro-data-files",
                str(tmp_path / "mmlu.jsonl"),
                "--truthfulqa-bin-path",
                "json",
                "--truthfulqa-bin-name",
                "null",
                "--truthfulqa-bin-split",
                "train",
                "--truthfulqa-bin-data-files",
                str(tmp_path / "truthful.jsonl"),
                "--arc-path",
                "json",
                "--arc-name",
                "null",
                "--arc-split",
                "train",
                "--arc-data-files",
                str(tmp_path / "arc.jsonl"),
                "--winogrande-path",
                "json",
                "--winogrande-name",
                "null",
                "--winogrande-split",
                "train",
                "--winogrande-data-files",
                str(tmp_path / "wino.jsonl"),
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

            manifest = json.loads((output_dir / "panel_manifest.json").read_text(encoding="utf-8"))
            stats = json.loads((output_dir / "panel_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["mmlu_pro"], 3)
            self.assertEqual(manifest["counts"]["truthfulqa_binary"], 2)
            self.assertEqual(stats["sources"]["mmlu_pro"]["excluded_due_overlap"], 1)
            self.assertEqual(stats["sources"]["truthfulqa_binary"]["excluded_due_overlap"], 1)

    @unittest.skipUnless(LM_EVAL_AVAILABLE, "lm_eval package is not installed")
    def test_utility_eval_and_merge_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            panel_dir = tmp_path / "utility_1k_v1"
            panel_dir.mkdir(parents=True)
            base_model_dir = tmp_path / "base_model"
            tokenizer_dir = tmp_path / "tokenizer"
            run_dir = tmp_path / "run"
            checkpoint_dir = run_dir / "checkpoint-2"
            eval_dir = run_dir / "evals"

            make_tokenizer(tokenizer_dir)
            make_tiny_llama_model(base_model_dir)
            run_dir.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)
            make_lora_adapter(base_model_dir, checkpoint_dir)
            make_lora_adapter(base_model_dir, run_dir)

            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 2, "epoch": 0.5}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 4, "epoch": 1.0}) + "\n",
                encoding="utf-8",
            )

            utility_examples = {
                "utility_mmlu_pro_400.jsonl": [
                    {
                        "id": "mmlu_pro:1",
                        "source": "mmlu_pro",
                        "category": "math",
                        "question": "sky blue",
                        "choices": ["A", "B", "C", "D"],
                        "gold_idx": 0,
                    }
                ],
                "utility_truthfulqa_bin_200.jsonl": [
                    {
                        "id": "truthfulqa_binary:1",
                        "source": "truthfulqa_binary",
                        "category": "truthfulness",
                        "question": "truth",
                        "choices": ["A", "B"],
                        "gold_idx": 0,
                    }
                ],
                "utility_arc_200.jsonl": [
                    {
                        "id": "arc_challenge:1",
                        "source": "arc_challenge",
                        "category": "science",
                        "question": "science planet",
                        "choices": ["A", "B", "C", "D"],
                        "gold_idx": 1,
                    }
                ],
                "utility_winogrande_200.jsonl": [
                    {
                        "id": "winogrande:1",
                        "source": "winogrande",
                        "category": "commonsense",
                        "question": "commonsense",
                        "choices": ["A", "B"],
                        "gold_idx": 1,
                    }
                ],
            }
            for filename, rows in utility_examples.items():
                write_jsonl(panel_dir / filename, rows)

            checkpoint_eval_dir = run_dir / "checkpoint_evals" / "checkpoint-2"
            checkpoint_eval_dir.mkdir(parents=True)
            (checkpoint_eval_dir / "DUET_SUMMARY.json").write_text(
                json.dumps({"forget_qa_rouge": 0.25, "holdout_qa_rouge": 0.75}) + "\n",
                encoding="utf-8",
            )
            (eval_dir / "DUET_SUMMARY.json").write_text(
                json.dumps({"forget_qa_rouge": 0.15, "holdout_qa_rouge": 0.8}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    str(PYTHON_BIN),
                    "src/tools/summarize_checkpoint_metrics.py",
                    "--run-dir",
                    str(run_dir),
                    "--output-path",
                    str(run_dir / "checkpoint_evals/summary.tsv"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            env = os.environ.copy()
            env["PATH"] = f"{REPO_ROOT / '.venv/bin'}:{env['PATH']}"
            env["UTILITY_ROOT"] = str(panel_dir)
            env["UTILITY_APPLY_CHAT_TEMPLATE"] = "false"
            env["UTILITY_EVAL_BATCH_SIZE"] = "1"
            subprocess.run(
                [
                    "bash",
                    "scripts/utility/eval_checkpoints_utility.sh",
                    str(run_dir),
                    "Llama-3.1-8B",
                    "Llama-3.1-8B-lora",
                    str(base_model_dir),
                    str(tokenizer_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )

            subprocess.run(
                [
                    str(PYTHON_BIN),
                    "src/tools/merge_checkpoint_utility_summaries.py",
                    "--checkpoint-summary",
                    str(run_dir / "checkpoint_evals/summary.tsv"),
                    "--utility-summary",
                    str(run_dir / "checkpoint_evals_utility/summary.tsv"),
                    "--output-path",
                    str(run_dir / "checkpoint_evals_merged/summary.tsv"),
                    "--trajectory-path",
                    str(run_dir / "checkpoint_evals_merged/trajectory_metrics.json"),
                    "--forget-tau",
                    "0.16",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            utility_summary = (run_dir / "checkpoint_evals_utility/summary.tsv").read_text(
                encoding="utf-8"
            )
            merged_summary = (run_dir / "checkpoint_evals_merged/summary.tsv").read_text(
                encoding="utf-8"
            )
            trajectory = json.loads(
                (run_dir / "checkpoint_evals_merged/trajectory_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertIn("base_model_orig", utility_summary)
            self.assertIn("checkpoint-2", utility_summary)
            self.assertIn("final", utility_summary)
            self.assertIn("utility_avg", merged_summary)
            self.assertEqual(trajectory["endpoint_label"], "final")
            self.assertEqual(trajectory["u_at_forget_tau"]["label"], "final")

    @unittest.skipUnless(LM_EVAL_AVAILABLE, "lm_eval package is not installed")
    def test_utility_eval_uses_last_checkpoint_as_final_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            panel_dir = tmp_path / "utility_1k_v1"
            panel_dir.mkdir(parents=True)
            base_model_dir = tmp_path / "base_model"
            tokenizer_dir = tmp_path / "tokenizer"
            run_dir = tmp_path / "run"
            checkpoint_dir = run_dir / "checkpoint-2"
            eval_dir = run_dir / "evals"

            make_tokenizer(tokenizer_dir)
            make_tiny_llama_model(base_model_dir)
            run_dir.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)
            make_lora_adapter(base_model_dir, checkpoint_dir)
            make_lora_adapter(base_model_dir, run_dir)
            (run_dir / "adapter_model.safetensors").unlink()

            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 2, "epoch": 0.5}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 4, "epoch": 1.0}) + "\n",
                encoding="utf-8",
            )

            utility_examples = {
                "utility_mmlu_pro_400.jsonl": [
                    {
                        "id": "mmlu_pro:1",
                        "source": "mmlu_pro",
                        "category": "math",
                        "question": "sky blue",
                        "choices": ["A", "B", "C", "D"],
                        "gold_idx": 0,
                    }
                ],
                "utility_truthfulqa_bin_200.jsonl": [
                    {
                        "id": "truthfulqa_binary:1",
                        "source": "truthfulqa_binary",
                        "category": "truthfulness",
                        "question": "truth",
                        "choices": ["A", "B"],
                        "gold_idx": 0,
                    }
                ],
                "utility_arc_200.jsonl": [
                    {
                        "id": "arc_challenge:1",
                        "source": "arc_challenge",
                        "category": "science",
                        "question": "science planet",
                        "choices": ["A", "B", "C", "D"],
                        "gold_idx": 1,
                    }
                ],
                "utility_winogrande_200.jsonl": [
                    {
                        "id": "winogrande:1",
                        "source": "winogrande",
                        "category": "commonsense",
                        "question": "commonsense",
                        "choices": ["A", "B"],
                        "gold_idx": 1,
                    }
                ],
            }
            for filename, rows in utility_examples.items():
                write_jsonl(panel_dir / filename, rows)

            checkpoint_eval_dir = run_dir / "checkpoint_evals" / "checkpoint-2"
            checkpoint_eval_dir.mkdir(parents=True)
            (checkpoint_eval_dir / "DUET_SUMMARY.json").write_text(
                json.dumps({"forget_qa_rouge": 0.25, "holdout_qa_rouge": 0.75}) + "\n",
                encoding="utf-8",
            )
            (eval_dir / "DUET_SUMMARY.json").write_text(
                json.dumps({"forget_qa_rouge": 0.15, "holdout_qa_rouge": 0.8}) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    str(PYTHON_BIN),
                    "src/tools/summarize_checkpoint_metrics.py",
                    "--run-dir",
                    str(run_dir),
                    "--output-path",
                    str(run_dir / "checkpoint_evals/summary.tsv"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            env = os.environ.copy()
            env["PATH"] = f"{REPO_ROOT / '.venv/bin'}:{env['PATH']}"
            env["UTILITY_ROOT"] = str(panel_dir)
            env["UTILITY_APPLY_CHAT_TEMPLATE"] = "false"
            env["UTILITY_EVAL_BATCH_SIZE"] = "1"
            subprocess.run(
                [
                    "bash",
                    "scripts/utility/eval_checkpoints_utility.sh",
                    str(run_dir),
                    "Llama-3.1-8B",
                    "Llama-3.1-8B-lora",
                    str(base_model_dir),
                    str(tokenizer_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )

            subprocess.run(
                [
                    str(PYTHON_BIN),
                    "src/tools/merge_checkpoint_utility_summaries.py",
                    "--checkpoint-summary",
                    str(run_dir / "checkpoint_evals/summary.tsv"),
                    "--utility-summary",
                    str(run_dir / "checkpoint_evals_utility/summary.tsv"),
                    "--output-path",
                    str(run_dir / "checkpoint_evals_merged/summary.tsv"),
                    "--trajectory-path",
                    str(run_dir / "checkpoint_evals_merged/trajectory_metrics.json"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            utility_summary = (run_dir / "checkpoint_evals_utility/summary.tsv").read_text(
                encoding="utf-8"
            )
            merged_summary = (run_dir / "checkpoint_evals_merged/summary.tsv").read_text(
                encoding="utf-8"
            )
            trajectory = json.loads(
                (run_dir / "checkpoint_evals_merged/trajectory_metrics.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertIn("final", utility_summary)
            self.assertIn("final", merged_summary)
            self.assertEqual(trajectory["endpoint_label"], "final")


if __name__ == "__main__":
    unittest.main()
