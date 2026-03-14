import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlTraceCallback(TrainerCallback):
    """Append trainer log records to an output-local JSONL trace."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        path = Path(args.output_dir) / "dualcf_trace.jsonl"
        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        record.update({str(key): value for key, value in logs.items()})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
