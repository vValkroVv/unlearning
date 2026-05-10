from trainer.utils import compute_undial_loss
from trainer.unlearn.grad_diff import GradDiff


class UNDIAL(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.uses_lora = hasattr(self.model, "disable_adapter")
        self._input_require_grads_handle = None
        if self.uses_lora:
            self._prepare_lora_for_gradient_checkpointing()
            self._ensure_trainable_lora_adapter()
        if self.ref_model is None and not self.uses_lora:
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_lora_for_gradient_checkpointing(self):
        enable_input_require_grads = getattr(
            self.model, "enable_input_require_grads", None
        )
        if callable(enable_input_require_grads):
            enable_input_require_grads()
            return

        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            return
        input_embeddings = get_input_embeddings()
        if input_embeddings is None:
            return

        def make_inputs_require_grad(_module, _inputs, output):
            output.requires_grad_(True)

        self._input_require_grads_handle = input_embeddings.register_forward_hook(
            make_inputs_require_grad
        )

    def _trainable_param_count(self):
        return sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )

    def _ensure_trainable_lora_adapter(self):
        if self._trainable_param_count() > 0:
            return

        active_adapter = getattr(self.model, "active_adapter", None)
        if callable(active_adapter):
            active_adapter = active_adapter()
        set_adapter = getattr(self.model, "set_adapter", None)
        if active_adapter is not None and callable(set_adapter):
            set_adapter(active_adapter)

        if self._trainable_param_count() == 0:
            raise ValueError(
                "UNDIAL LoRA training found no trainable adapter parameters. "
                "If loading an existing adapter, make sure it is loaded as trainable."
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = compute_undial_loss(
            model, self.ref_model, forget_inputs, self.beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        if model.training and not loss.requires_grad:
            raise RuntimeError(
                "UNDIAL loss has no gradient path. This usually means LoRA input "
                "gradients or trainable adapter parameters are disabled."
            )
        return (loss, forget_outputs) if return_outputs else loss
