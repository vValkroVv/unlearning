from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss_dynamic_beta


class PopDynamBWGA(GradDiff):
    def __init__(self, rep_coeff: float = 0.0, *args, **kwargs):
        # Accept and ignore any provided 'beta' to stay compatible with WGA experiment presets
        if "beta" in kwargs:
            kwargs.pop("beta", None)
        super().__init__(*args, **kwargs)
        self.rep_coeff = rep_coeff
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        # Keep pop_sum if present; the dynamic beta function will use it
        forget_inputs = {
            k: forget_inputs[k]
            for k in ("input_ids", "attention_mask", "labels", "pop_sum")
            if k in forget_inputs
        }
        forget_loss, forget_outputs = compute_wga_loss_dynamic_beta(
            model=model,
            inputs=forget_inputs,
            beta_from_pop_sum=True,
            rep_coeff=self.rep_coeff,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
