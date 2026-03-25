from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class SaveOnEpochsCallback(TrainerCallback):
    """Request checkpoint saves when training crosses specific epoch values."""

    def __init__(self, save_epochs):
        self.save_epochs = sorted({float(epoch) for epoch in (save_epochs or [])})
        self._done = set()

    def _mark_completed_from_existing_checkpoints(self, output_dir: str) -> None:
        checkpoint_root = Path(output_dir)
        if not checkpoint_root.exists():
            return

        for checkpoint_dir in sorted(checkpoint_root.glob("checkpoint-*")):
            state_path = checkpoint_dir / "trainer_state.json"
            if not state_path.exists():
                continue

            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            epoch = payload.get("epoch")
            if epoch is None:
                continue

            epoch = float(epoch)
            for target_epoch in self.save_epochs:
                if epoch + 1e-8 >= target_epoch:
                    self._done.add(round(target_epoch, 8))

    def on_train_begin(self, args, state, control, **kwargs):
        self._done.clear()

        # Only infer completed targets from existing checkpoints when Trainer
        # state already indicates a resumed run. Fresh FORCE_RERUN retrains can
        # reuse the same output dir, so blindly scanning checkpoint-* would
        # incorrectly suppress the new intermediate save.
        if state.global_step > 0 or float(state.epoch or 0.0) > 0.0:
            self._mark_completed_from_existing_checkpoints(args.output_dir)

        current_epoch = float(state.epoch or 0.0)
        for epoch in self.save_epochs:
            if current_epoch + 1e-8 >= epoch:
                self._done.add(round(epoch, 8))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return control

        current_epoch = float(state.epoch)
        final_epoch = float(args.num_train_epochs)

        for epoch in self.save_epochs:
            epoch_key = round(epoch, 8)

            # Keep the top-level endpoint save on the normal trainer.save_model() path.
            if epoch >= final_epoch - 1e-8:
                continue

            if epoch_key in self._done:
                continue

            if current_epoch + 1e-8 >= epoch:
                control.should_save = True
                self._done.add(epoch_key)

        return control
