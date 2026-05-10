import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    compute_nll_per_sample,
    compute_npo_per_sample,
    compute_weighted_nll_per_sample,
    compute_weighted_npo_per_sample,
)


@dataclass
class _SAMState:
    e_ws: List[Optional[torch.Tensor]]


class GeneralCF(GradDiff):
    """
    General counterfactual unlearning family.

    Supported forget-side decomposition:
        gamma * (lambda_cf * CE(counterfactual)
               + lambda_add * L_additional)
      + lambda_ret * CE(retain)

    where L_additional is one of:
        - EMPTY   : disable the additional branch (lambda_additional = 0)
        - CE      : -CE(original)
        - NPO     : NPO(original, ref)
        - NPO-SAM : SAM applied only to the additional branch

    Routing modes:
        - full           : current DualCF-style routing
        - d_only         : difficulty-only routing
        - a_only         : attribution-only routing
        - constant       : one global constant triplet estimated from reference artifacts
        - constant_split : one split-local constant triplet estimated from the current artifact

    Span controls:
        - span_additional: apply span weighting to the additional branch
        - span_cf_branch : optional legacy switch; when true, also apply span
                           weighting to the counterfactual branch, which makes the
                           trainer backward-compatible with current SpanCF losses.
    """

    LOG_PREFIX = "generalcf"

    def __init__(
        self,
        beta=0.5,
        tau_d=0.5,
        tau_a=0.5,
        temp_d=0.25,
        temp_a=0.25,
        lambda_neg_max=1.0,
        lambda_ret_lo=1.0,
        lambda_ret_hi=2.0,
        cf_weight=1.0,
        risk_forget_scale=0.5,
        normalize_cf_by_tokens=True,
        normalize_neg_by_tokens=True,
        disable_difficulty_route=False,
        disable_attribution_route=False,
        rarity_neg_gain=0.0,
        rarity_cf_gain=0.0,
        disable_rarity_route=False,
        alpha_eff_stat="topk_mean",
        alpha_eff_topk_frac=0.25,
        risk_power=1.0,
        neg_power=1.0,
        additional_loss="NPO",
        span_additional=False,
        span_cf_branch=False,
        routing_mode="full",
        span_mode="lcs",
        alt_shared_token_weight=0.0,
        alt_unique_token_weight=1.0,
        orig_shared_token_weight=0.0,
        orig_unique_token_weight=1.0,
        cf_branch_scale=1.0,
        additional_branch_scale=1.0,
        neg_branch_scale=None,
        sam_rho=0.01,
        sam_adaptive=False,
        sam_eps=1e-12,
        lambda_cf_const=None,
        lambda_additional_const=None,
        lambda_retain_const=None,
        constant_artifact_paths=None,
        constant_current_artifact_path=None,
        constant_batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self.tau_d = float(tau_d)
        self.tau_a = float(tau_a)
        self.temp_d = float(temp_d)
        self.temp_a = float(temp_a)
        self.lambda_neg_max = float(lambda_neg_max)
        self.lambda_ret_lo = float(lambda_ret_lo)
        self.lambda_ret_hi = float(lambda_ret_hi)
        self.cf_weight = float(cf_weight)
        self.risk_forget_scale = float(risk_forget_scale)
        self.normalize_cf_by_tokens = bool(normalize_cf_by_tokens)
        self.normalize_neg_by_tokens = bool(normalize_neg_by_tokens)
        self.disable_difficulty_route = bool(disable_difficulty_route)
        self.disable_attribution_route = bool(disable_attribution_route)
        self.rarity_neg_gain = float(rarity_neg_gain)
        self.rarity_cf_gain = float(rarity_cf_gain)
        self.disable_rarity_route = bool(disable_rarity_route)
        self.alpha_eff_stat = str(alpha_eff_stat)
        self.alpha_eff_topk_frac = float(alpha_eff_topk_frac)
        self.risk_power = float(risk_power)
        self.neg_power = float(neg_power)

        self.additional_loss = self._normalize_additional_loss(additional_loss)
        self.span_additional = bool(span_additional)
        self.span_cf_branch = bool(span_cf_branch)
        self.routing_mode = self._normalize_routing_mode(routing_mode)
        self.span_mode = str(span_mode)
        self.alt_shared_token_weight = float(alt_shared_token_weight)
        self.alt_unique_token_weight = float(alt_unique_token_weight)
        self.orig_shared_token_weight = float(orig_shared_token_weight)
        self.orig_unique_token_weight = float(orig_unique_token_weight)
        self.cf_branch_scale = float(cf_branch_scale)
        if neg_branch_scale is not None and additional_branch_scale == 1.0:
            additional_branch_scale = neg_branch_scale
        self.additional_branch_scale = float(additional_branch_scale)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        self.sam_eps = float(sam_eps)

        self.lambda_cf_const = self._optional_float(lambda_cf_const)
        self.lambda_additional_const = self._optional_float(lambda_additional_const)
        self.lambda_retain_const = self._optional_float(lambda_retain_const)
        self.constant_artifact_paths = self._parse_artifact_paths(constant_artifact_paths)
        self.constant_current_artifact_path = self._parse_artifact_paths(
            constant_current_artifact_path
        )
        self.constant_batch_size = (
            None
            if constant_batch_size in {None, "", "null", "None"}
            else int(constant_batch_size)
        )
        self.log_prefix = str(getattr(self, "LOG_PREFIX", "generalcf"))
        self._constant_lambda_cache = None

        if self.beta <= 0.0:
            raise ValueError("GeneralCF requires beta > 0.")
        if self.temp_d <= 0.0 or self.temp_a <= 0.0:
            raise ValueError("GeneralCF requires temp_d > 0 and temp_a > 0.")
        if self.alpha_eff_topk_frac <= 0.0 or self.alpha_eff_topk_frac > 1.0:
            raise ValueError("GeneralCF requires 0 < alpha_eff_topk_frac <= 1.")
        if self.span_mode not in {"lcs", "set_overlap"}:
            raise ValueError("GeneralCF span_mode must be `lcs` or `set_overlap`.")
        if min(
            self.alt_shared_token_weight,
            self.alt_unique_token_weight,
            self.orig_shared_token_weight,
            self.orig_unique_token_weight,
        ) < 0.0:
            raise ValueError("GeneralCF token weights must be non-negative.")
        if self.cf_branch_scale < 0.0 or self.additional_branch_scale < 0.0:
            raise ValueError("GeneralCF branch scales must be non-negative.")

        if self.routing_mode == "d_only":
            self.disable_attribution_route = True
        elif self.routing_mode == "a_only":
            self.disable_difficulty_route = True

        needs_ref_model = self.retain_loss_type == "KL" or self.additional_loss in {
            "NPO",
            "NPO_SAM",
        }
        if needs_ref_model and self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    @staticmethod
    def _optional_float(value):
        if value in {None, "", "null", "None"}:
            return None
        return float(value)

    @staticmethod
    def _parse_artifact_paths(value) -> list[str]:
        if value in {None, "", "null", "None"}:
            return []
        if isinstance(value, (list, tuple)):
            return [
                str(item)
                for item in value
                if str(item) not in {"", "null", "None"}
            ]
        text = str(value)
        for sep in ("::", "|", ";", ","):
            if sep in text:
                return [
                    part
                    for part in (item.strip() for item in text.split(sep))
                    if part and part not in {"null", "None"}
                ]
        return [text]

    @staticmethod
    def _normalize_additional_loss(value: str) -> str:
        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "NONE": "EMPTY",
            "NULL": "EMPTY",
            "NO": "EMPTY",
            "NPO_SAM": "NPO_SAM",
            "NPOSAM": "NPO_SAM",
            "SAMNPO": "NPO_SAM",
            "CE": "CE",
            "NPO": "NPO",
            "EMPTY": "EMPTY",
        }
        if normalized not in aliases:
            raise ValueError(
                "GeneralCF additional_loss must be one of EMPTY, CE, NPO, NPO-SAM. "
                f"Got {value!r}."
            )
        return aliases[normalized]

    @staticmethod
    def _normalize_routing_mode(value: str) -> str:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "full": "full",
            "d_only": "d_only",
            "difficulty_only": "d_only",
            "a_only": "a_only",
            "attribution_only": "a_only",
            "constant": "constant",
            "constant_split": "constant_split",
            "split_constant": "constant_split",
        }
        if normalized not in aliases:
            raise ValueError(
                "GeneralCF routing_mode must be one of full, d_only, a_only, "
                f"constant, constant_split. Got {value!r}."
            )
        return aliases[normalized]

    def _retain_inputs(self, inputs):
        retain_inputs = inputs["retain"]
        return {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

    def _score_tensor(self, forget_inputs, key: str, device, batch_size: int):
        if key not in forget_inputs:
            raise KeyError(
                f"GeneralCF expected `inputs['forget']['{key}']` but it was missing. "
                "Check the forget dataset and collator metadata plumbing."
            )
        score = forget_inputs[key]
        if torch.is_tensor(score):
            score = score.to(device=device, dtype=torch.float32)
        else:
            score = torch.tensor(score, device=device, dtype=torch.float32)
        score = score.view(-1)
        if score.numel() != batch_size:
            raise ValueError(
                f"GeneralCF expected `{key}` to have {batch_size} values, got "
                f"{score.numel()}."
            )
        return score

    def _optional_score_tensor(self, forget_inputs, key: str, device, batch_size: int):
        if key not in forget_inputs:
            return None
        return self._score_tensor(
            forget_inputs=forget_inputs,
            key=key,
            device=device,
            batch_size=batch_size,
        )

    def _summarize_risk(self, risk_gate: torch.Tensor) -> torch.Tensor:
        if self.alpha_eff_stat == "mean":
            return risk_gate.mean()
        if self.alpha_eff_stat == "p75":
            return torch.quantile(risk_gate, 0.75)
        if self.alpha_eff_stat == "max":
            return risk_gate.max()
        if self.alpha_eff_stat == "topk_mean":
            topk = max(1, int(math.ceil(risk_gate.numel() * self.alpha_eff_topk_frac)))
            return torch.topk(risk_gate, k=topk).values.mean()
        raise ValueError(f"Unknown alpha_eff_stat={self.alpha_eff_stat}")

    def _compute_routing_from_scores(
        self,
        difficulty: torch.Tensor,
        attribution: torch.Tensor,
        rarity: torch.Tensor,
    ) -> dict:
        if self.disable_difficulty_route:
            difficulty_gate = torch.ones_like(difficulty)
        else:
            difficulty_gate = torch.sigmoid((difficulty - self.tau_d) / self.temp_d)
            difficulty_gate = difficulty_gate.pow(self.neg_power)

        if self.disable_attribution_route:
            risk_gate = torch.zeros_like(attribution)
        else:
            risk_gate = torch.sigmoid((attribution - self.tau_a) / self.temp_a)
            risk_gate = risk_gate.pow(self.risk_power)

        lambda_neg_base = self.lambda_neg_max * difficulty_gate * (1.0 - risk_gate)
        lambda_neg = lambda_neg_base * (1.0 + self.rarity_neg_gain * rarity)
        cf_weight_eff = self.cf_weight * (1.0 - self.rarity_cf_gain * rarity)
        cf_weight_eff = cf_weight_eff.clamp_min(0.0)
        forget_scale = 1.0 - (1.0 - self.risk_forget_scale) * risk_gate

        return {
            "difficulty": difficulty,
            "attribution": attribution,
            "rarity": rarity,
            "difficulty_gate": difficulty_gate,
            "risk_gate": risk_gate,
            "lambda_neg_base": lambda_neg_base,
            "lambda_neg": lambda_neg,
            "cf_weight_eff": cf_weight_eff,
            "forget_scale": forget_scale,
        }

    def _compute_routing_full(self, forget_inputs, batch_size: int, device):
        difficulty = self._score_tensor(
            forget_inputs, "difficulty_score", device=device, batch_size=batch_size
        )
        attribution = self._score_tensor(
            forget_inputs, "attribution_score", device=device, batch_size=batch_size
        )
        rarity = self._optional_score_tensor(
            forget_inputs, "rarity_score", device=device, batch_size=batch_size
        )
        if rarity is None or self.disable_rarity_route:
            rarity = torch.zeros_like(difficulty)
        else:
            rarity = rarity.clamp(0.0, 1.0)

        routing = self._compute_routing_from_scores(
            difficulty=difficulty,
            attribution=attribution,
            rarity=rarity,
        )
        risk_batch = self._summarize_risk(routing["risk_gate"])
        lambda_ret_batch = self.lambda_ret_lo + (
            self.lambda_ret_hi - self.lambda_ret_lo
        ) * risk_batch
        alpha_eff = self.alpha * lambda_ret_batch
        routing.update(
            {
                "risk_batch": risk_batch,
                "lambda_ret_batch": lambda_ret_batch,
                "alpha_eff": alpha_eff,
                "routing_mode": self.routing_mode,
            }
        )
        return routing

    def _artifact_paths_for_constant_routing(self) -> list[str]:
        if self.routing_mode == "constant_split":
            if self.constant_current_artifact_path:
                return list(self.constant_current_artifact_path)
            if self.constant_artifact_paths:
                return [self.constant_artifact_paths[0]]
        else:
            if self.constant_artifact_paths:
                return list(self.constant_artifact_paths)
            if self.constant_current_artifact_path:
                return list(self.constant_current_artifact_path)
        return []

    def _iter_artifact_rows(self, paths: Sequence[str]) -> Iterable[dict]:
        for raw_path in paths:
            path = Path(str(raw_path)).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"GeneralCF constant routing artifact not found: {path}"
                )
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

    def _estimate_constant_lambdas_from_rows(self, rows: Sequence[dict]) -> dict:
        if not rows:
            raise ValueError(
                "GeneralCF constant routing could not load any rows from the reference artifact."
            )

        difficulty = torch.tensor(
            [float(row["difficulty_score"]) for row in rows],
            dtype=torch.float32,
        )
        attribution = torch.tensor(
            [float(row["attribution_score"]) for row in rows],
            dtype=torch.float32,
        )
        rarity = torch.tensor(
            [float(row.get("rarity_score", 0.0)) for row in rows],
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)
        if self.disable_rarity_route:
            rarity.zero_()

        routed = self._compute_routing_from_scores(
            difficulty=difficulty,
            attribution=attribution,
            rarity=rarity,
        )

        lambda_cf_samples = routed["forget_scale"] * routed["cf_weight_eff"]
        lambda_additional_samples = routed["forget_scale"] * routed["lambda_neg"]

        batch_size = int(
            self.constant_batch_size or self.args.per_device_train_batch_size or 1
        )
        weighted_alpha_total = 0.0
        weighted_ret_total = 0.0
        total_rows = 0

        for start in range(0, int(routed["risk_gate"].numel()), batch_size):
            end = min(start + batch_size, int(routed["risk_gate"].numel()))
            chunk = routed["risk_gate"][start:end]
            chunk_size = max(end - start, 1)

            risk_batch = self._summarize_risk(chunk)
            lambda_ret_batch = self.lambda_ret_lo + (
                self.lambda_ret_hi - self.lambda_ret_lo
            ) * risk_batch
            alpha_eff_batch = self.alpha * lambda_ret_batch

            weighted_ret_total += float(lambda_ret_batch.detach().item()) * chunk_size
            weighted_alpha_total += float(alpha_eff_batch.detach().item()) * chunk_size
            total_rows += chunk_size

        return {
            "lambda_cf": float(lambda_cf_samples.mean().detach().item()),
            "lambda_additional": float(
                lambda_additional_samples.mean().detach().item()
            ),
            "lambda_ret": float(weighted_alpha_total / max(total_rows, 1)),
            "lambda_ret_batch": float(weighted_ret_total / max(total_rows, 1)),
            "num_rows": len(rows),
        }

    def _estimate_constant_lambdas(self) -> dict:
        if (
            self.lambda_cf_const is not None
            and self.lambda_additional_const is not None
            and self.lambda_retain_const is not None
        ):
            return {
                "lambda_cf": float(self.lambda_cf_const),
                "lambda_additional": float(self.lambda_additional_const),
                "lambda_ret": float(self.lambda_retain_const),
                "source": "manual",
                "num_files": 0,
                "num_rows": 0,
            }

        if self._constant_lambda_cache is not None:
            return self._constant_lambda_cache

        artifact_paths = self._artifact_paths_for_constant_routing()
        if not artifact_paths:
            raise ValueError(
                "GeneralCF constant routing requires either explicit lambda_*_const "
                "values, constant_artifact_paths, or constant_current_artifact_path."
            )

        per_file = []
        total_rows = 0

        for raw_path in artifact_paths:
            rows = list(self._iter_artifact_rows([raw_path]))
            if not rows:
                continue
            stats = self._estimate_constant_lambdas_from_rows(rows)
            stats["path"] = str(raw_path)
            per_file.append(stats)
            total_rows += int(stats["num_rows"])

        if not per_file:
            raise ValueError(
                "GeneralCF constant routing could not load any rows from the reference artifacts."
            )

        num_files = len(per_file)
        constants = {
            "lambda_cf": float(sum(item["lambda_cf"] for item in per_file) / num_files),
            "lambda_additional": float(
                sum(item["lambda_additional"] for item in per_file) / num_files
            ),
            "lambda_ret": float(
                sum(item["lambda_ret"] for item in per_file) / num_files
            ),
            "lambda_ret_batch": float(
                sum(item["lambda_ret_batch"] for item in per_file) / num_files
            ),
            "source": (
                "artifact_single_split_mean"
                if num_files == 1
                else "artifact_equal_split_mean"
            ),
            "num_files": num_files,
            "num_rows": total_rows,
            "per_file": per_file,
        }
        self._constant_lambda_cache = constants
        return constants

    def _compute_routing_constant(self, forget_inputs, batch_size: int, device):
        difficulty = self._score_tensor(
            forget_inputs, "difficulty_score", device=device, batch_size=batch_size
        )
        attribution = self._score_tensor(
            forget_inputs, "attribution_score", device=device, batch_size=batch_size
        )
        rarity = self._optional_score_tensor(
            forget_inputs, "rarity_score", device=device, batch_size=batch_size
        )
        if rarity is None or self.disable_rarity_route:
            rarity = torch.zeros_like(difficulty)
        else:
            rarity = rarity.clamp(0.0, 1.0)

        constants = self._estimate_constant_lambdas()
        lambda_cf = float(constants["lambda_cf"])
        lambda_additional = float(constants["lambda_additional"])
        lambda_ret = float(constants["lambda_ret"])
        lambda_ret_batch = float(
            constants.get("lambda_ret_batch", lambda_ret / max(float(self.alpha), 1e-8))
        )
        return {
            "difficulty": difficulty,
            "attribution": attribution,
            "rarity": rarity,
            "difficulty_gate": torch.ones_like(difficulty),
            "risk_gate": torch.zeros_like(difficulty),
            "lambda_neg_base": torch.full_like(difficulty, lambda_additional),
            "lambda_neg": torch.full_like(difficulty, lambda_additional),
            "cf_weight_eff": torch.full_like(difficulty, lambda_cf),
            "forget_scale": torch.ones_like(difficulty),
            "risk_batch": torch.tensor(0.0, device=device, dtype=torch.float32),
            "lambda_ret_batch": torch.tensor(
                lambda_ret_batch,
                device=device,
                dtype=torch.float32,
            ),
            "alpha_eff": torch.tensor(lambda_ret, device=device, dtype=torch.float32),
            "routing_mode": self.routing_mode,
            "constant_routing": True,
            "constant_num_rows": float(constants.get("num_rows", 0)),
            "constant_num_files": float(constants.get("num_files", 0)),
        }

    def _compute_routing(self, forget_inputs, batch_size: int, device):
        if self.routing_mode in {"full", "d_only", "a_only"}:
            routing = self._compute_routing_full(
                forget_inputs=forget_inputs,
                batch_size=batch_size,
                device=device,
            )
        elif self.routing_mode in {"constant", "constant_split"}:
            routing = self._compute_routing_constant(
                forget_inputs=forget_inputs,
                batch_size=batch_size,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported routing_mode={self.routing_mode}")

        if self.additional_loss == "EMPTY":
            routing = dict(routing)
            zero = torch.zeros_like(routing["lambda_neg"])
            routing["lambda_neg_base"] = zero
            routing["lambda_neg"] = zero

        return routing

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
            unique_fractions.append(
                float(token_count - shared_count) / float(token_count)
            )

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

    def _compute_cf_term(self, model, forget_inputs, original_inputs):
        alternate_inputs = forget_inputs["alternate"]
        if self.span_cf_branch:
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

        cf_vec, outputs = compute_nll_per_sample(
            model,
            alternate_inputs,
            normalize_by_tokens=self.normalize_cf_by_tokens,
        )
        return cf_vec, outputs, {}

    def _compute_additional_term(self, model, original_inputs, alternate_inputs):
        if self.additional_loss == "EMPTY":
            zeros = torch.zeros(
                original_inputs["input_ids"].shape[0],
                device=original_inputs["input_ids"].device,
                dtype=torch.float32,
            )
            return zeros, {}

        span_stats = {}
        if self.span_additional:
            add_weights, span_stats = self._build_token_diff_weights(
                original_inputs["labels"],
                alternate_inputs["labels"],
                shared_weight=self.orig_shared_token_weight,
                unique_weight=self.orig_unique_token_weight,
            )
            if self.additional_loss == "CE":
                add_vec, _ = compute_weighted_nll_per_sample(
                    model,
                    original_inputs,
                    token_weights=add_weights,
                )
                return -add_vec, span_stats
            add_vec, _ = compute_weighted_npo_per_sample(
                model,
                self.ref_model,
                original_inputs,
                token_weights=add_weights,
                beta=self.beta,
            )
            return add_vec, span_stats

        if self.additional_loss == "CE":
            add_vec, _ = compute_nll_per_sample(
                model,
                original_inputs,
                normalize_by_tokens=self.normalize_neg_by_tokens,
            )
            return -add_vec, span_stats

        add_vec, _ = compute_npo_per_sample(
            model,
            self.ref_model,
            original_inputs,
            beta=self.beta,
            normalize_by_tokens=self.normalize_neg_by_tokens,
        )
        return add_vec, span_stats

    def _forget_term_components(
        self,
        cf_vec: torch.Tensor,
        additional_vec: torch.Tensor,
        routing: dict,
    ) -> dict:
        cf_weight_eff = routing.get("cf_weight_eff", self.cf_weight)
        per_sample_cf_loss = routing["forget_scale"] * (cf_weight_eff * cf_vec)
        per_sample_additional_loss = routing["forget_scale"] * (
            routing["lambda_neg"] * additional_vec
        )
        per_sample_forget_loss = per_sample_cf_loss + per_sample_additional_loss
        return {
            "per_sample_cf_loss": per_sample_cf_loss,
            "per_sample_additional_loss": per_sample_additional_loss,
            "per_sample_forget_loss": per_sample_forget_loss,
            "cf_loss": per_sample_cf_loss.mean(),
            "additional_loss": per_sample_additional_loss.mean(),
            "forget_loss": per_sample_forget_loss.mean(),
            # Legacy aliases for downstream logs/comparison code.
            "per_sample_neg_loss": per_sample_additional_loss,
            "neg_loss": per_sample_additional_loss.mean(),
        }

    def _additional_log_suffix(self) -> str:
        return {
            "EMPTY": "empty",
            "CE": "ce",
            "NPO": "npo",
            "NPO_SAM": "npo_sam",
        }[self.additional_loss]

    def _compute_core_components(self, model, inputs):
        forget_inputs = inputs["forget"]
        original_inputs = forget_inputs["original"]
        alternate_inputs = forget_inputs["alternate"]

        cf_vec, outputs, cf_logs = self._compute_cf_term(
            model=model,
            forget_inputs=forget_inputs,
            original_inputs=original_inputs,
        )
        additional_vec, additional_logs = self._compute_additional_term(
            model=model,
            original_inputs=original_inputs,
            alternate_inputs=alternate_inputs,
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
            additional_vec=additional_vec,
            routing=routing,
        )

        extra_logs = {
            f"{self.log_prefix}_routing_mode_full": (
                1.0 if self.routing_mode == "full" else 0.0
            ),
            f"{self.log_prefix}_routing_mode_d_only": (
                1.0 if self.routing_mode == "d_only" else 0.0
            ),
            f"{self.log_prefix}_routing_mode_a_only": (
                1.0 if self.routing_mode == "a_only" else 0.0
            ),
            f"{self.log_prefix}_routing_mode_constant": (
                1.0 if self.routing_mode == "constant" else 0.0
            ),
            f"{self.log_prefix}_routing_mode_constant_split": (
                1.0 if self.routing_mode == "constant_split" else 0.0
            ),
            f"{self.log_prefix}_additional_empty": (
                1.0 if self.additional_loss == "EMPTY" else 0.0
            ),
            f"{self.log_prefix}_additional_ce": (
                1.0 if self.additional_loss == "CE" else 0.0
            ),
            f"{self.log_prefix}_additional_npo": (
                1.0 if self.additional_loss == "NPO" else 0.0
            ),
            f"{self.log_prefix}_additional_npo_sam": (
                1.0 if self.additional_loss == "NPO_SAM" else 0.0
            ),
            f"{self.log_prefix}_span_additional": (
                1.0 if self.span_additional else 0.0
            ),
            f"{self.log_prefix}_span_cf_branch": (
                1.0 if self.span_cf_branch else 0.0
            ),
        }
        extra_logs.update(cf_logs)
        if additional_logs:
            extra_logs.update(
                {
                    f"{self.log_prefix}_additional_shared_token_frac": additional_logs[
                        "shared_fraction_mean"
                    ],
                    f"{self.log_prefix}_additional_unique_token_frac": additional_logs[
                        "unique_fraction_mean"
                    ],
                }
            )
        if routing.get("constant_routing"):
            extra_logs.update(
                {
                    f"{self.log_prefix}_constant_num_rows": float(
                        routing.get("constant_num_rows", 0.0)
                    ),
                    f"{self.log_prefix}_constant_num_files": float(
                        routing.get("constant_num_files", 0.0)
                    ),
                }
            )

        return {
            "forget_inputs": forget_inputs,
            "original_inputs": original_inputs,
            "alternate_inputs": alternate_inputs,
            "cf_vec": cf_vec,
            "additional_vec": additional_vec,
            "outputs": outputs,
            **routing,
            **forget_terms,
            "extra_logs": extra_logs,
        }

    def _build_log_payload(self, components: dict, retain_loss, loss=None, extra_logs=None):
        prefix = self.log_prefix
        alpha_eff = components["alpha_eff"]
        alpha_eff_value = (
            float(alpha_eff.detach().item())
            if torch.is_tensor(alpha_eff)
            else float(alpha_eff)
        )
        payload = {
            f"{prefix}_cf_loss": float(components["cf_vec"].mean().detach().item()),
            f"{prefix}_additional_loss": float(
                components["additional_vec"].mean().detach().item()
            ),
            f"{prefix}_neg_loss": float(
                components["additional_vec"].mean().detach().item()
            ),
            f"{prefix}_forget_loss": float(components["forget_loss"].detach().item()),
            f"{prefix}_retain_loss": float(retain_loss.detach().item()),
            f"{prefix}_alpha_eff": alpha_eff_value,
            f"{prefix}_lambda_ret_batch": float(
                components["lambda_ret_batch"].detach().item()
            ),
            f"{prefix}_d_mean": float(components["difficulty"].mean().detach().item()),
            f"{prefix}_a_mean": float(
                components["attribution"].mean().detach().item()
            ),
            f"{prefix}_u_mean": float(components["rarity"].mean().detach().item()),
            f"{prefix}_u_p50": float(
                torch.quantile(components["rarity"], 0.50).detach().item()
            ),
            f"{prefix}_u_p90": float(
                torch.quantile(components["rarity"], 0.90).detach().item()
            ),
            f"{prefix}_s_mean": float(
                components["difficulty_gate"].mean().detach().item()
            ),
            f"{prefix}_s_p50": float(
                torch.quantile(components["difficulty_gate"], 0.50).detach().item()
            ),
            f"{prefix}_s_p90": float(
                torch.quantile(components["difficulty_gate"], 0.90).detach().item()
            ),
            f"{prefix}_r_mean": float(components["risk_gate"].mean().detach().item()),
            f"{prefix}_r_p50": float(
                torch.quantile(components["risk_gate"], 0.50).detach().item()
            ),
            f"{prefix}_r_p90": float(
                torch.quantile(components["risk_gate"], 0.90).detach().item()
            ),
            f"{prefix}_risk_batch": float(components["risk_batch"].detach().item()),
            f"{prefix}_lambda_additional_mean": float(
                components["lambda_neg"].mean().detach().item()
            ),
            f"{prefix}_lambda_neg_mean": float(
                components["lambda_neg"].mean().detach().item()
            ),
            f"{prefix}_lambda_neg_base_mean": float(
                components["lambda_neg_base"].mean().detach().item()
            ),
            f"{prefix}_lambda_cf_mean": float(
                components["cf_weight_eff"].mean().detach().item()
            ),
            f"{prefix}_cf_weight_eff_mean": float(
                components["cf_weight_eff"].mean().detach().item()
            ),
        }
        if loss is not None:
            payload[f"{prefix}_total_loss"] = float(loss.detach().item())
        payload.update(components.get("extra_logs", {}))
        if extra_logs:
            payload.update(extra_logs)

        logged_cf_loss = components.get("logged_cf_loss", None)
        logged_additional_loss = components.get("logged_additional_loss", None)
        logged_neg_loss = components.get("logged_neg_loss", None)
        logged_forget_loss = components.get("logged_forget_loss", None)

        if logged_cf_loss is not None:
            payload[f"{prefix}_cf_loss"] = float(logged_cf_loss.detach().item())
        if logged_additional_loss is not None:
            payload[f"{prefix}_additional_loss"] = float(
                logged_additional_loss.detach().item()
            )
        if logged_neg_loss is not None:
            payload[f"{prefix}_neg_loss"] = float(logged_neg_loss.detach().item())
        if logged_forget_loss is not None:
            payload[f"{prefix}_forget_loss"] = float(
                logged_forget_loss.detach().item()
            )
        return payload

    def _maybe_log(self, components: dict, retain_loss, loss=None, extra_logs=None):
        try:
            self.log(
                self._build_log_payload(
                    components=components,
                    retain_loss=retain_loss,
                    loss=loss,
                    extra_logs=extra_logs,
                )
            )
        except Exception:
            pass

    def compute_loss(self, model, inputs, return_outputs=False):
        components = self._compute_core_components(model, inputs)
        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )
        loss = self.gamma * components["forget_loss"] + components["alpha_eff"] * retain_loss
        self._maybe_log(components=components, retain_loss=retain_loss, loss=loss)
        return (loss, components["outputs"]) if return_outputs else loss

    def _trainable_params(self, model: torch.nn.Module) -> List[torch.nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    def _stash_grads(self, params: List[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
        stashed: List[Optional[torch.Tensor]] = []
        for param in params:
            if param.grad is None:
                stashed.append(None)
            else:
                stashed.append(param.grad.detach().clone())
        return stashed

    def _clear_grads_set_to_none(self, params: List[torch.nn.Parameter]) -> None:
        for param in params:
            param.grad = None

    def _set_final_grads(
        self,
        params: List[torch.nn.Parameter],
        second_pass_grads: Sequence[Optional[torch.Tensor]],
        prev_grads: List[Optional[torch.Tensor]],
        grad_scale: float,
    ) -> None:
        for param, grad_second, grad_prev in zip(params, second_pass_grads, prev_grads):
            grad = None
            if grad_second is not None:
                grad = grad_second.detach() * grad_scale
            if grad_prev is not None:
                grad = grad_prev if grad is None else grad + grad_prev
            param.grad = grad

    @torch.no_grad()
    def _grad_norm(
        self,
        params: List[torch.nn.Parameter],
        grads: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        if not params:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        ref_device = None
        for grad in grads:
            if grad is not None:
                ref_device = grad.device
                break
        if ref_device is None:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        sq_sum = torch.zeros((), device=ref_device, dtype=torch.float32)
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            grad_term = grad
            if self.sam_adaptive:
                grad_term = param.detach().abs() * grad_term
            grad_sq = grad_term.float()
            if grad_sq.device != ref_device:
                grad_sq = grad_sq.to(ref_device)
            sq_sum = sq_sum + (grad_sq * grad_sq).sum()
        return torch.sqrt(sq_sum)

    @torch.no_grad()
    def _perturb_weights(
        self,
        params: List[torch.nn.Parameter],
        grads: Sequence[Optional[torch.Tensor]],
        grad_norm: torch.Tensor,
    ) -> _SAMState:
        scale = self.sam_rho / (grad_norm + self.sam_eps)
        e_ws: List[Optional[torch.Tensor]] = []

        for param, grad in zip(params, grads):
            if grad is None:
                e_ws.append(None)
                continue

            perturb = param.detach().abs() * grad if self.sam_adaptive else grad
            scale_t = scale.to(device=perturb.device, dtype=perturb.dtype)
            e_w = (perturb * scale_t).to(dtype=param.dtype)
            param.add_(e_w)
            e_ws.append(e_w)

        return _SAMState(e_ws=e_ws)

    @torch.no_grad()
    def _restore_weights(self, params: List[torch.nn.Parameter], state: _SAMState) -> None:
        for param, e_w in zip(params, state.e_ws):
            if e_w is None:
                continue
            param.sub_(e_w)

    def _apply_branch_scales(
        self,
        cf_loss: torch.Tensor,
        additional_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cf_branch_scale * cf_loss,
            self.additional_branch_scale * additional_loss,
        )

    def _compute_additional_loss_only(self, model, inputs):
        components = self._compute_core_components(model, inputs)
        self._last_manual_components = components
        return components["additional_loss"]

    def _compute_retain_loss_only(self, model, inputs):
        return self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )

    def _manual_retain_weight(self, components=None):
        if components is None:
            return self.alpha
        return components["alpha_eff"]

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.additional_loss != "NPO_SAM":
            return super().training_step(model, inputs)

        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[GeneralCF] DeepSpeed is not supported in the NPO-SAM manual-grad path."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[GeneralCF] Multi-process training is not supported in the NPO-SAM "
                "manual-grad path."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        params = self._trainable_params(model)
        if not params:
            raise RuntimeError("[GeneralCF] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps
        self._last_manual_components = None

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

        with self.compute_loss_context_manager():
            components = self._compute_core_components(model, inputs)
            cf_loss_base = components["cf_loss"]
            additional_loss_1_base = components["additional_loss"]
            cf_loss, additional_loss_1 = self._apply_branch_scales(
                cf_loss=cf_loss_base,
                additional_loss=additional_loss_1_base,
            )

        cf_grads = torch.autograd.grad(
            cf_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        additional_grads_1 = torch.autograd.grad(
            additional_loss_1,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_norm = self._grad_norm(params, additional_grads_1)
        sam_state = self._perturb_weights(params, additional_grads_1, grad_norm)

        try:
            self._clear_grads_set_to_none(params)
            with self.compute_loss_context_manager():
                additional_loss_2_base = self._compute_additional_loss_only(model, inputs)
                additional_loss_2 = (
                    self.additional_branch_scale * additional_loss_2_base
                )
            additional_grads_2 = torch.autograd.grad(
                additional_loss_2,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            self._restore_weights(params, sam_state)

        self._clear_grads_set_to_none(params)
        with self.compute_loss_context_manager():
            retain_loss = self._compute_retain_loss_only(model, inputs)
        retain_grads = torch.autograd.grad(
            retain_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        retain_weight = self._manual_retain_weight(components=components)
        retain_weight_value = (
            float(retain_weight.detach().item())
            if torch.is_tensor(retain_weight)
            else float(retain_weight)
        )

        combined_grads = []
        for grad_cf, grad_add, grad_retain in zip(
            cf_grads, additional_grads_2, retain_grads
        ):
            grad = None
            if grad_cf is not None:
                grad = self.gamma * grad_cf
            if grad_add is not None:
                add_component = self.gamma * grad_add
                grad = add_component if grad is None else grad + add_component
            if grad_retain is not None:
                retain_scale = (
                    retain_weight.to(
                        device=grad_retain.device,
                        dtype=grad_retain.dtype,
                    )
                    if torch.is_tensor(retain_weight)
                    else retain_weight
                )
                retain_component = retain_scale * grad_retain
                grad = retain_component if grad is None else grad + retain_component
            combined_grads.append(grad)

        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)

        retain_weight_loss = (
            retain_weight.to(device=retain_loss.device, dtype=retain_loss.dtype)
            if torch.is_tensor(retain_weight)
            else retain_weight
        )
        forget_loss_update = cf_loss + additional_loss_2
        total_loss = self.gamma * forget_loss_update + retain_weight_loss * retain_loss

        self._last_manual_components = components
        components["logged_cf_loss"] = cf_loss
        components["logged_additional_loss"] = additional_loss_2
        components["logged_neg_loss"] = additional_loss_2
        components["logged_forget_loss"] = forget_loss_update
        extra_logs = {
            f"{self.log_prefix}_cf_loss_base": float(cf_loss_base.detach().item()),
            f"{self.log_prefix}_cf_loss_scaled": float(cf_loss.detach().item()),
            f"{self.log_prefix}_additional_loss_1_base": float(
                additional_loss_1_base.detach().item()
            ),
            f"{self.log_prefix}_additional_loss_1_scaled": float(
                additional_loss_1.detach().item()
            ),
            f"{self.log_prefix}_additional_loss_2_base": float(
                additional_loss_2_base.detach().item()
            ),
            f"{self.log_prefix}_additional_loss_2_scaled": float(
                additional_loss_2.detach().item()
            ),
            f"{self.log_prefix}_retain_loss": float(retain_loss.detach().item()),
            f"{self.log_prefix}_retain_weight": retain_weight_value,
            f"{self.log_prefix}_cf_branch_scale": float(self.cf_branch_scale),
            f"{self.log_prefix}_additional_branch_scale": float(
                self.additional_branch_scale
            ),
            f"{self.log_prefix}_grad_norm": float(grad_norm.detach().item()),
            f"{self.log_prefix}_rho": float(self.sam_rho),
            f"{self.log_prefix}_adaptive": 1.0 if self.sam_adaptive else 0.0,
        }
        try:
            self._maybe_log(
                components=components,
                retain_loss=retain_loss,
                loss=total_loss,
                extra_logs=extra_logs,
            )
        except Exception:
            pass

        return total_loss.detach() * grad_scale
