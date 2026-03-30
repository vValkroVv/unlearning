from trainer.unlearn.npo import NPO
from trainer.unlearn.sam_mixin import SAMMixin
from trainer.utils import compute_dpo_loss


class NPOSAM(SAMMixin, NPO):
    """
    NPO + SAM (Sharpness-Aware Minimization), as used in Unlearn-Smooth.

    SAM is implemented with a two-pass update:
    1) compute gradients at current weights, then perturb parameters by +e(w)
    2) compute gradients at perturbed weights (actual update gradients)
    3) restore original weights
    """

    def __init__(
        self,
        sam_rho: float = 0.01,
        sam_adaptive: bool = False,
        sam_eps: float = 1e-12,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        self.sam_eps = float(sam_eps)

    SAM_LOG_PREFIX = "npo_sam"

    def _compute_forget_loss_only(self, model, inputs):
        forget_inputs = inputs["forget"]
        if isinstance(forget_inputs, dict) and "original" in forget_inputs:
            forget_inputs = forget_inputs["original"]

        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        return forget_loss

    def _compute_retain_loss_only(self, model, inputs):
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
