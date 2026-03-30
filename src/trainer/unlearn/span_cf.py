import torch

from trainer.unlearn.dual_cf import DualCF
from trainer.utils import (
    compute_weighted_nll_per_sample,
    compute_weighted_npo_per_sample,
)


class SpanCF(DualCF):
    LOG_PREFIX = "span"

    def __init__(
        self,
        span_mode="lcs",
        alt_shared_token_weight=None,
        alt_unique_token_weight=None,
        orig_shared_token_weight=None,
        orig_unique_token_weight=None,
        shared_token_weight=None,
        unique_token_weight=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.span_mode = str(span_mode)

        legacy_shared = (
            float(shared_token_weight) if shared_token_weight is not None else None
        )
        legacy_unique = (
            float(unique_token_weight) if unique_token_weight is not None else None
        )
        self.alt_shared_token_weight = float(
            0.25
            if alt_shared_token_weight is None and legacy_shared is None
            else (
                legacy_shared
                if alt_shared_token_weight is None
                else alt_shared_token_weight
            )
        )
        self.alt_unique_token_weight = float(
            1.0
            if alt_unique_token_weight is None and legacy_unique is None
            else (
                legacy_unique
                if alt_unique_token_weight is None
                else alt_unique_token_weight
            )
        )
        self.orig_shared_token_weight = float(
            self.alt_shared_token_weight
            if orig_shared_token_weight is None
            else orig_shared_token_weight
        )
        self.orig_unique_token_weight = float(
            self.alt_unique_token_weight
            if orig_unique_token_weight is None
            else orig_unique_token_weight
        )
        self.shared_token_weight = self.alt_shared_token_weight
        self.unique_token_weight = self.alt_unique_token_weight

        if self.span_mode not in {"lcs", "set_overlap"}:
            raise ValueError("SpanCF span_mode must be `lcs` or `set_overlap`.")
        if min(
            self.alt_shared_token_weight,
            self.alt_unique_token_weight,
            self.orig_shared_token_weight,
            self.orig_unique_token_weight,
        ) < 0.0:
            raise ValueError("SpanCF token weights must be non-negative.")

    def _shared_positions(self, tokens: list[int], other_tokens: list[int]) -> set[int]:
        if self.span_mode == "set_overlap":
            other_set = set(other_tokens)
            return {idx for idx, token in enumerate(tokens) if token in other_set}

        rows = len(tokens)
        cols = len(other_tokens)
        dp = [[0] * (cols + 1) for _ in range(rows + 1)]
        for row_idx in range(rows):
            for col_idx in range(cols):
                if tokens[row_idx] == other_tokens[col_idx]:
                    dp[row_idx + 1][col_idx + 1] = dp[row_idx][col_idx] + 1
                else:
                    dp[row_idx + 1][col_idx + 1] = max(
                        dp[row_idx][col_idx + 1],
                        dp[row_idx + 1][col_idx],
                    )

        shared = set()
        row_idx = rows
        col_idx = cols
        while row_idx > 0 and col_idx > 0:
            if tokens[row_idx - 1] == other_tokens[col_idx - 1]:
                shared.add(row_idx - 1)
                row_idx -= 1
                col_idx -= 1
            elif dp[row_idx - 1][col_idx] >= dp[row_idx][col_idx - 1]:
                row_idx -= 1
            else:
                col_idx -= 1
        return shared

    def _build_token_diff_weights(
        self,
        labels: torch.Tensor,
        other_labels: torch.Tensor,
        shared_weight: float,
        unique_weight: float,
    ):
        weights = torch.zeros(labels.shape, device=labels.device, dtype=torch.float32)
        shared_fractions = []
        unique_fractions = []

        for batch_idx in range(labels.shape[0]):
            valid_mask = labels[batch_idx] != -100
            other_valid_mask = other_labels[batch_idx] != -100
            tokens = labels[batch_idx][valid_mask].detach().tolist()
            other_tokens = other_labels[batch_idx][other_valid_mask].detach().tolist()
            if not tokens:
                shared_fractions.append(0.0)
                unique_fractions.append(0.0)
                continue

            shared_positions = self._shared_positions(tokens, other_tokens)
            row_weights = []
            for token_idx in range(len(tokens)):
                row_weights.append(
                    shared_weight if token_idx in shared_positions else unique_weight
                )

            weights[batch_idx, valid_mask] = torch.tensor(
                row_weights,
                device=labels.device,
                dtype=torch.float32,
            )
            shared_count = len(shared_positions)
            token_count = max(len(tokens), 1)
            shared_fractions.append(float(shared_count) / float(token_count))
            unique_fractions.append(float(token_count - shared_count) / float(token_count))

        return weights, {
            "shared_fraction_mean": (
                sum(shared_fractions) / float(len(shared_fractions))
                if shared_fractions
                else 0.0
            ),
            "unique_fraction_mean": (
                sum(unique_fractions) / float(len(unique_fractions))
                if unique_fractions
                else 0.0
            ),
        }

    def _compute_cf_vec(self, model, original_inputs, alternate_inputs):
        cf_weights, cf_stats = self._build_token_diff_weights(
            alternate_inputs["labels"],
            original_inputs["labels"],
            shared_weight=self.alt_shared_token_weight,
            unique_weight=self.alt_unique_token_weight,
        )
        cf_vec, outputs = compute_weighted_nll_per_sample(
            model,
            alternate_inputs,
            token_weights=cf_weights,
        )
        return cf_vec, outputs, cf_stats

    def _compute_neg_vec(self, model, original_inputs, alternate_inputs):
        neg_weights, neg_stats = self._build_token_diff_weights(
            original_inputs["labels"],
            alternate_inputs["labels"],
            shared_weight=self.orig_shared_token_weight,
            unique_weight=self.orig_unique_token_weight,
        )
        neg_vec, _ = compute_weighted_npo_per_sample(
            model,
            self.ref_model,
            original_inputs,
            token_weights=neg_weights,
            beta=self.beta,
        )
        return neg_vec, neg_stats

    def _compute_core_components(self, model, inputs):
        forget_inputs = inputs["forget"]
        original_inputs = forget_inputs["original"]
        alternate_inputs = forget_inputs["alternate"]

        cf_vec, outputs, cf_stats = self._compute_cf_vec(
            model,
            original_inputs,
            alternate_inputs,
        )
        neg_vec, neg_stats = self._compute_neg_vec(
            model,
            original_inputs,
            alternate_inputs,
        )

        batch_size = int(cf_vec.shape[0])
        device = cf_vec.device
        routing = self._compute_routing(
            forget_inputs=forget_inputs,
            batch_size=batch_size,
            device=device,
        )
        forget_terms = self._forget_term_components(
            cf_vec=cf_vec,
            neg_vec=neg_vec,
            routing=routing,
            forget_inputs=forget_inputs,
        )

        components = {
            "forget_inputs": forget_inputs,
            "original_inputs": original_inputs,
            "cf_vec": cf_vec,
            "neg_vec": neg_vec,
            "outputs": outputs,
            **routing,
            **forget_terms,
            "extra_logs": {
                f"{self.log_prefix}_alt_shared_token_frac": cf_stats["shared_fraction_mean"],
                f"{self.log_prefix}_alt_unique_token_frac": cf_stats["unique_fraction_mean"],
                f"{self.log_prefix}_orig_shared_token_frac": neg_stats["shared_fraction_mean"],
                f"{self.log_prefix}_orig_unique_token_frac": neg_stats["unique_fraction_mean"],
            },
        }
        components["extra_logs"].update(self._extra_log_components(components))
        return components
