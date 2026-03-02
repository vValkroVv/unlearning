import logging
import os
from typing import Optional

from model.fila import apply_fila_initialization
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import ihl_loss_from_logits

logger = logging.getLogger(__name__)


class LoKU(GradDiff):
    """LoKU: Inverted Hinge forget objective + FILA initialization for LoRA."""

    def __init__(
        self,
        ihl_alpha: float = 1.0,
        importance_file: Optional[str] = None,
        fila_eps: float = 1e-5,
        fila_adapter_name: str = "default",
        fila_base_subdir: str = "base_model",
        run_fila_sanity_check: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ihl_alpha = float(ihl_alpha)
        self.importance_file = importance_file
        self.fila_eps = float(fila_eps)
        self.fila_adapter_name = str(fila_adapter_name)
        self.fila_base_subdir = str(fila_base_subdir)
        self.run_fila_sanity_check = bool(run_fila_sanity_check)

        self._fila_enabled = False
        self._fila_applied = False
        self._fila_stats = None
        self._fila_base_saved = False

        if self.importance_file not in (None, "", "null", "None"):
            self._fila_enabled = True
            self._apply_fila_or_raise()

    def _as_model_inputs(self, batch):
        if isinstance(batch, dict) and "original" in batch:
            batch = batch["original"]
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _resolve_lora_targets_and_rank(self):
        peft_config = getattr(self.model, "peft_config", None)
        if peft_config is None:
            raise ValueError(
                "LoKU with FILA requires a LoRA/PEFT model, but no peft_config was found."
            )

        if not isinstance(peft_config, dict) or not peft_config:
            raise ValueError("Invalid peft_config structure on model.")

        cfg = peft_config.get(self.fila_adapter_name)
        if cfg is None:
            cfg = next(iter(peft_config.values()))
            logger.warning(
                "[LoKU] Adapter '%s' not found; using first available adapter config.",
                self.fila_adapter_name,
            )

        target_modules = list(getattr(cfg, "target_modules", []) or [])
        if not target_modules:
            raise ValueError("LoKU FILA needs non-empty model.lora_config.target_modules.")

        rank = int(getattr(cfg, "r", 0))
        if rank <= 0:
            raise ValueError("LoKU FILA requires LoRA rank `r` > 0.")

        return target_modules, rank

    def _apply_fila_or_raise(self):
        importance_file = str(self.importance_file)
        if not os.path.exists(importance_file):
            raise FileNotFoundError(
                f"[LoKU] importance_file does not exist: {importance_file}"
            )

        target_modules, lora_rank = self._resolve_lora_targets_and_rank()
        logger.info(
            "[LoKU] Applying FILA from %s (targets=%s, rank=%d, eps=%g, adapter=%s)",
            importance_file,
            target_modules,
            lora_rank,
            self.fila_eps,
            self.fila_adapter_name,
        )
        self._fila_stats = apply_fila_initialization(
            peft_model=self.model,
            importance_file=importance_file,
            target_modules=target_modules,
            lora_rank=lora_rank,
            eps=self.fila_eps,
            adapter_name=self.fila_adapter_name,
            strict=True,
            run_sanity_check=self.run_fila_sanity_check,
        )
        self._fila_applied = True

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._as_model_inputs(inputs["forget"])
        forget_outputs = model(**forget_inputs)
        forget_loss = ihl_loss_from_logits(
            logits=forget_outputs.logits,
            labels=forget_inputs["labels"],
            alpha=self.ihl_alpha,
        )

        retain_inputs = self._as_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def _save_fila_base_model(self, output_dir: str) -> None:
        if self._fila_base_saved:
            return
        if not self.is_world_process_zero():
            return

        base_dir = os.path.join(output_dir, self.fila_base_subdir)
        os.makedirs(base_dir, exist_ok=True)

        model_to_save = self.model
        if getattr(self, "accelerator", None) is not None:
            try:
                model_to_save = self.accelerator.unwrap_model(self.model)
            except Exception:
                model_to_save = self.model

        if hasattr(model_to_save, "unload") and callable(model_to_save.unload):
            base_model = model_to_save.unload()
        else:
            raise ValueError(
                "[LoKU] FILA base save requires a PEFT model with `unload()` support."
            )

        base_model.save_pretrained(base_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(base_dir)

        self._fila_base_saved = True
        logger.info("[LoKU] Saved FILA residual base model to %s", base_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        out_dir = output_dir or self.args.output_dir
        result = super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if not self._fila_enabled:
            return result
        if not self._fila_applied:
            raise ValueError("[LoKU] FILA is enabled but was not successfully applied.")
        if _internal_call:
            return result
        if "checkpoint-" in str(out_dir):
            return result

        self._save_fila_base_model(output_dir=out_dir)
        return result
