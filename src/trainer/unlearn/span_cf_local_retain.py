from trainer.unlearn.span_cf import SpanCF
from trainer.unlearn.span_cf_simnpo import SpanCFSimNPO


class _SpanCFLocalRetainMixin:
    def __init__(
        self,
        local_retain_weight=0.2,
        boundary_margin_weight=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.local_retain_weight = float(local_retain_weight)
        self.boundary_margin_weight = float(boundary_margin_weight)

    def _local_retain_inputs(self, inputs):
        local_retain_inputs = inputs["forget"]["local_retain"]
        return {
            "input_ids": local_retain_inputs["input_ids"],
            "attention_mask": local_retain_inputs["attention_mask"],
            "labels": local_retain_inputs["labels"],
        }

    def _optional_boundary_score_tensor(self, forget_inputs, keys, device, batch_size):
        for key in keys:
            value = self._optional_score_tensor(
                forget_inputs=forget_inputs,
                key=key,
                device=device,
                batch_size=batch_size,
            )
            if value is not None:
                return value
        return None

    def _forget_term_components(self, cf_vec, neg_vec, routing, forget_inputs):
        payload = super()._forget_term_components(
            cf_vec=cf_vec,
            neg_vec=neg_vec,
            routing=routing,
            forget_inputs=forget_inputs,
        )
        boundary_score = self._optional_score_tensor(
            forget_inputs=forget_inputs,
            key="boundary_score",
            device=cf_vec.device,
            batch_size=int(cf_vec.shape[0]),
        )
        if boundary_score is None or self.boundary_margin_weight == 0.0:
            return payload

        margin = 1.0 + self.boundary_margin_weight * boundary_score.clamp_min(0.0)
        payload["per_sample_cf_loss"] = payload["per_sample_cf_loss"] * margin
        payload["per_sample_neg_loss"] = payload["per_sample_neg_loss"] * margin
        payload["per_sample_forget_loss"] = (
            payload["per_sample_cf_loss"] + payload["per_sample_neg_loss"]
        )
        payload["cf_loss"] = payload["per_sample_cf_loss"].mean()
        payload["neg_loss"] = payload["per_sample_neg_loss"].mean()
        payload["forget_loss"] = payload["per_sample_forget_loss"].mean()
        return payload

    def _extra_log_components(self, components: dict) -> dict:
        payload = super()._extra_log_components(components)
        forget_inputs = components["forget_inputs"]
        batch_size = int(components["cf_vec"].shape[0])
        device = components["cf_vec"].device
        boundary_score = self._optional_score_tensor(
            forget_inputs=forget_inputs,
            key="boundary_score",
            device=device,
            batch_size=batch_size,
        )
        relation = self._optional_boundary_score_tensor(
            forget_inputs=forget_inputs,
            keys=("boundary_relation", "boundary_relation_score"),
            device=device,
            batch_size=batch_size,
        )
        overlap = self._optional_boundary_score_tensor(
            forget_inputs=forget_inputs,
            keys=("boundary_overlap", "boundary_lexical_overlap"),
            device=device,
            batch_size=batch_size,
        )
        if boundary_score is not None:
            payload[f"{self.log_prefix}_score_mean"] = float(
                boundary_score.mean().detach().item()
            )
            payload[f"{self.log_prefix}_margin_factor_mean"] = float(
                (
                    1.0 + self.boundary_margin_weight * boundary_score.clamp_min(0.0)
                ).mean().detach().item()
            )
        if relation is not None:
            payload[f"{self.log_prefix}_relation_mean"] = float(
                relation.mean().detach().item()
            )
        if overlap is not None:
            payload[f"{self.log_prefix}_overlap_mean"] = float(
                overlap.mean().detach().item()
            )
        return payload

    def compute_loss(self, model, inputs, return_outputs=False):
        components = self._compute_core_components(model, inputs)
        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )
        local_retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=self._local_retain_inputs(inputs),
        )
        loss = (
            self.gamma * components["forget_loss"]
            + components["alpha_eff"] * retain_loss
            + self.local_retain_weight * local_retain_loss
        )
        self._maybe_log(
            components=components,
            retain_loss=retain_loss,
            loss=loss,
            extra_logs={
                f"{self.log_prefix}_local_retain_loss": float(
                    local_retain_loss.detach().item()
                ),
                f"{self.log_prefix}_local_retain_weight": self.local_retain_weight,
            },
        )
        return (loss, components["outputs"]) if return_outputs else loss


class SpanCFLocalRetain(_SpanCFLocalRetainMixin, SpanCF):
    LOG_PREFIX = "span_local"


class SpanCFSimNPOLocalRetain(_SpanCFLocalRetainMixin, SpanCFSimNPO):
    LOG_PREFIX = "span_simnpo_local"
