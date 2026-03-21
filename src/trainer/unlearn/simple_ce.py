from trainer.unlearn.grad_diff import GradDiff


class SimpleCE(GradDiff):
    """
    Simple counterfactual CE baseline.

    Objective:
        loss = gamma * (-CE(forget.original))
             + cf_weight * CE(forget.alternate)
             + retain_weight * CE(retain)

    This reuses the existing counterfactual artifact format produced for
    DualCF / DPO, but removes routing and preference terms.
    """

    def __init__(self, cf_weight=1.0, retain_weight=1.0, *args, **kwargs):
        # Keep `alpha` as a backward-compatible alias for older configs/scripts.
        alpha = kwargs.pop("alpha", None)
        if alpha is not None and retain_weight == 1.0:
            retain_weight = alpha
        super().__init__(alpha=retain_weight, *args, **kwargs)
        self.cf_weight = float(cf_weight)
        self.retain_weight = float(retain_weight)

    @staticmethod
    def _as_model_inputs(batch):
        if isinstance(batch, dict) and "original" in batch:
            batch = batch["original"]
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._as_model_inputs(inputs["forget"]["original"])
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        cf_inputs = self._as_model_inputs(inputs["forget"]["alternate"])
        cf_outputs = model(**cf_inputs)
        cf_loss = cf_outputs.loss

        retain_inputs = self._as_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.cf_weight * cf_loss + self.retain_weight * retain_loss

        try:
            self.log(
                {
                    "simple_ce_forget_loss": float(forget_loss.detach().item()),
                    "simple_ce_cf_loss": float(cf_loss.detach().item()),
                    "simple_ce_retain_loss": float(retain_loss.detach().item()),
                    "simple_ce_cf_weight": self.cf_weight,
                    "simple_ce_retain_weight": self.retain_weight,
                    "simple_ce_total_loss": float(loss.detach().item()),
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss
