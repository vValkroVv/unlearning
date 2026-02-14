import json
from pathlib import Path


def parse_task_name(name: str):
    if "duet_" in name:
        base = name[name.index("duet_") :]
    else:
        return None
    if "_lora_" not in base:
        return None
    prefix, suffix = base.split("_lora_", 1)
    prefix_parts = prefix.split("_")
    if len(prefix_parts) < 4:
        return None
    model = prefix_parts[1]
    trainer = prefix_parts[-1]
    forget = "_".join(prefix_parts[2:-1])

    key_vals = {}
    for chunk in suffix.split("_"):
        if chunk.startswith("r") and "=" not in chunk:
            key = "lora_r"
            value = chunk[1:]
        elif chunk.startswith("lalpha"):
            key = "lora_alpha"
            value = chunk[len("lalpha") :]
        elif chunk.startswith("alpha"):
            key = "lora_alpha"
            value = chunk[len("alpha") :]
        elif chunk.startswith("ldrop"):
            key = "lora_drop"
            value = chunk[len("ldrop") :]
        elif chunk.startswith("drop"):
            key = "lora_drop"
            value = chunk[len("drop") :]
        elif chunk.startswith("lr"):
            key = "lr"
            value = chunk[len("lr") :]
        elif chunk.startswith("beta"):
            key = "beta"
            value = chunk[len("beta") :]
        elif chunk.startswith("malpha"):
            key = "alpha"
            value = chunk[len("malpha") :]
        else:
            key = None
            value = None
        if key:
            key_vals[key] = value

    if "lora_r" not in key_vals or "lr" not in key_vals or "beta" not in key_vals:
        return None

    # normalize tags (convert p to decimal)
    def norm(val: str):
        return val.replace("p", ".")

    method_alpha = key_vals.get("alpha")
    params = {
        "model": model,
        "forget_split": forget,
        "trainer": trainer,
        "lora_r": int(float(norm(key_vals["lora_r"]))),
        "lora_alpha": int(float(norm(key_vals.get("lora_alpha", "0")))),
        "lora_dropout": float(norm(key_vals.get("lora_drop", "0"))),
        "learning_rate": float(norm(key_vals["lr"])),
        "beta": float(norm(key_vals["beta"])),
        "alpha": float(norm(method_alpha)) if method_alpha is not None else None,
    }
    return params


def main():
    root = Path("saves/unlearn")
    rows = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        params = parse_task_name(task_dir.name)
        if not params:
            continue
        eval_dir = task_dir / "evals"
        summary_path = eval_dir / "DUET_SUMMARY.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        forget_score = summary.get("forget_qa_rouge")
        holdout_score = summary.get("holdout_qa_rouge")
        # Derive runtime from train logs if present
        train_runtime = None
        train_info_path = task_dir / "train_runtime.json"
        if train_info_path.exists():
            with train_info_path.open() as f:
                train_runtime = json.load(f).get("train_runtime")
        else:
            # fall back to parse TrainerState json if available
            trainer_state = task_dir / "trainer_state.json"
            if trainer_state.exists():
                with trainer_state.open() as f:
                    state = json.load(f)
                metrics = state.get("log_history", [])
                for entry in reversed(metrics):
                    if "train_runtime" in entry:
                        train_runtime = entry["train_runtime"]
                        break

        rows.append(
            {
                **params,
                "forget_qa_rouge": forget_score,
                "holdout_qa_rouge": holdout_score,
                "train_runtime_sec": train_runtime,
                "summary_path": str(summary_path),
            }
        )

    rows.sort(
        key=lambda r: (r["forget_split"], r["learning_rate"], r["beta"], r["alpha"])
    )

    headers = [
        "forget_split",
        "learning_rate",
        "beta",
        "alpha",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "forget_qa_rouge",
        "holdout_qa_rouge",
        "train_runtime_sec",
        "summary_path",
    ]

    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(h, "")) for h in headers))


if __name__ == "__main__":
    main()
