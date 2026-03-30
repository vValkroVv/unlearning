from trainer.unlearn.span_cf import SpanCF
from trainer.utils import compute_weighted_simnpo_per_sample


class SpanCFSimNPO(SpanCF):
    LOG_PREFIX = "span_simnpo"

    def __init__(self, delta=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = float(delta)

    def _compute_neg_vec(self, model, original_inputs, alternate_inputs):
        neg_weights, neg_stats = self._build_token_diff_weights(
            original_inputs["labels"],
            alternate_inputs["labels"],
            shared_weight=self.orig_shared_token_weight,
            unique_weight=self.orig_unique_token_weight,
        )
        neg_vec, _ = compute_weighted_simnpo_per_sample(
            model,
            original_inputs,
            token_weights=neg_weights,
            beta=self.beta,
            delta=self.delta,
        )
        return neg_vec, neg_stats
