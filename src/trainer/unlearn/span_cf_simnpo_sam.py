from trainer.unlearn.sam_mixin import SAMMixin
from trainer.unlearn.span_cf_simnpo import SpanCFSimNPO


class SpanCFSimNPOSAM(SAMMixin, SpanCFSimNPO):
    LOG_PREFIX = "span_simnpo_sam"
    SAM_LOG_PREFIX = "span_simnpo_sam"

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

    def _compute_forget_loss_only(self, model, inputs):
        components = self._compute_core_components(model, inputs)
        self._last_manual_components = components
        return components["forget_loss"]

    def _compute_retain_loss_only(self, model, inputs):
        return self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )

    def _manual_retain_weight(self, inputs, components=None):
        del inputs
        if components is None:
            return self.alpha
        return components["alpha_eff"]
