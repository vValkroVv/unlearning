import torch

from trainer.unlearn.dual_cf import DualCF
from trainer.utils import compute_nll_per_sample


class MultiCF(DualCF):
    LOG_PREFIX = "multicf"

    def __init__(
        self,
        max_alternates_used=4,
        alt_agg_mode="weighted_mean",
        alt_weight_mode="rerank",
        alt_set_temperature=0.7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_alternates_used = int(max_alternates_used)
        self.alt_agg_mode = str(alt_agg_mode)
        self.alt_weight_mode = str(alt_weight_mode)
        self.alt_set_temperature = float(alt_set_temperature)

        if self.max_alternates_used <= 0:
            raise ValueError("MultiCF requires max_alternates_used > 0.")
        if self.alt_agg_mode not in {"weighted_mean", "mean", "top1"}:
            raise ValueError(
                "MultiCF alt_agg_mode must be one of weighted_mean, mean, or top1."
            )
        if self.alt_weight_mode not in {"rerank", "uniform"}:
            raise ValueError(
                "MultiCF alt_weight_mode must be one of rerank or uniform."
            )
        if self.alt_set_temperature <= 0.0:
            raise ValueError("MultiCF requires alt_set_temperature > 0.")

    def _normalize_alt_weights(self, raw_weights: torch.Tensor, alt_mask: torch.Tensor):
        alt_mask = alt_mask.to(dtype=torch.bool)
        if self.alt_agg_mode == "top1":
            weights = torch.zeros_like(raw_weights)
            first_valid = alt_mask.float().argmax(dim=1)
            valid_rows = alt_mask.any(dim=1)
            if valid_rows.any():
                row_ids = torch.arange(raw_weights.shape[0], device=raw_weights.device)
                weights[row_ids[valid_rows], first_valid[valid_rows]] = 1.0
            return weights

        if self.alt_agg_mode == "mean" or self.alt_weight_mode == "uniform":
            weights = alt_mask.float()
        else:
            weights = raw_weights.clamp_min(0.0)
            if self.alt_set_temperature != 1.0:
                weights = torch.pow(weights.clamp_min(1e-8), 1.0 / self.alt_set_temperature)

        weights = weights * alt_mask.float()
        valid_rows = alt_mask.any(dim=1)
        if valid_rows.any():
            zero_rows = valid_rows & (weights.sum(dim=1) <= 0.0)
            if zero_rows.any():
                fallback = alt_mask[zero_rows].float()
                fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                weights[zero_rows] = fallback
        weights = weights * alt_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights / denom

    def _compute_cf_term(self, model, forget_inputs):
        if "alternates" not in forget_inputs:
            raise KeyError(
                "MultiCF expected `inputs['forget']['alternates']` from "
                "QAMultiCFDataset/DataCollatorForMultiCF."
            )

        all_alternates = list(forget_inputs["alternates"])
        if not all_alternates:
            raise ValueError("MultiCF received an empty alternates list.")

        alt_mask = forget_inputs["alternate_mask"].to(dtype=torch.bool)
        raw_weights = forget_inputs["alternate_weights"].to(dtype=torch.float32)
        max_used = min(len(all_alternates), alt_mask.shape[1], self.max_alternates_used)
        alternates = all_alternates[:max_used]
        alt_mask = alt_mask[:, :max_used]
        raw_weights = raw_weights[:, :max_used]

        cf_stack = []
        outputs = None
        for alternate_inputs in alternates:
            cf_vec, outputs = compute_nll_per_sample(
                model,
                alternate_inputs,
                normalize_by_tokens=self.normalize_cf_by_tokens,
            )
            cf_stack.append(cf_vec)

        cf_stack = torch.stack(cf_stack, dim=1)
        weights = self._normalize_alt_weights(raw_weights, alt_mask)
        cf_vec = (cf_stack * weights).sum(dim=1)

        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=1)
        payload = {
            f"{self.log_prefix}_num_alts_mean": float(
                alt_mask.float().sum(dim=1).mean().detach().item()
            ),
            f"{self.log_prefix}_weight_entropy": float(entropy.mean().detach().item()),
            f"{self.log_prefix}_top1_share": float(weights[:, 0].mean().detach().item()),
        }
        return cf_vec, outputs, payload
