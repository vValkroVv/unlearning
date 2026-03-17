import math

import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    build_answer_minus_shared_mask,
    build_answer_only_mask,
    compute_nll_per_sample,
    compute_npo_per_sample,
    compute_npo_per_sample_masked,
)


class DualCF(GradDiff):
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
        alpha_eff_stat="topk_mean",
        alpha_eff_topk_frac=0.25,
        risk_power=1.0,
        neg_power=1.0,
        belief_neg_weight=0.0,
        belief_beta=None,
        local_neg_mode="full",
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
        self.alpha_eff_stat = str(alpha_eff_stat)
        self.alpha_eff_topk_frac = float(alpha_eff_topk_frac)
        self.risk_power = float(risk_power)
        self.neg_power = float(neg_power)
        self.belief_neg_weight = float(belief_neg_weight)
        self.belief_beta = (
            None if belief_beta in (None, "", "null", "None") else float(belief_beta)
        )
        self.local_neg_mode = str(local_neg_mode)

        if self.beta <= 0.0:
            raise ValueError("DualCF requires beta > 0.")
        if self.belief_beta is not None and self.belief_beta <= 0.0:
            raise ValueError("DualCF requires belief_beta > 0 when provided.")
        if self.temp_d <= 0.0 or self.temp_a <= 0.0:
            raise ValueError("DualCF requires temp_d > 0 and temp_a > 0.")
        if self.alpha_eff_topk_frac <= 0.0 or self.alpha_eff_topk_frac > 1.0:
            raise ValueError("DualCF requires 0 < alpha_eff_topk_frac <= 1.")
        if self.local_neg_mode not in {
            "full",
            "answer_only",
            "answer_minus_shared",
        }:
            raise ValueError(
                "DualCF requires local_neg_mode in {'full', 'answer_only', 'answer_minus_shared'}."
            )

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

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
                f"DualCF expected `inputs['forget']['{key}']` but it was missing. "
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
                f"DualCF expected `{key}` to have {batch_size} values, got "
                f"{score.numel()}."
            )
        return score

    def _routing_state(self, forget_inputs, device, batch_size: int):
        difficulty = self._score_tensor(
            forget_inputs, "difficulty_score", device=device, batch_size=batch_size
        )
        attribution = self._score_tensor(
            forget_inputs, "attribution_score", device=device, batch_size=batch_size
        )

        difficulty_gate = torch.sigmoid((difficulty - self.tau_d) / self.temp_d)
        risk_gate = torch.sigmoid((attribution - self.tau_a) / self.temp_a)
        difficulty_gate = difficulty_gate.pow(self.neg_power)
        risk_gate = risk_gate.pow(self.risk_power)
        if self.disable_difficulty_route:
            difficulty_gate = torch.ones_like(difficulty_gate)
        if self.disable_attribution_route:
            risk_gate = torch.zeros_like(risk_gate)
        return difficulty, attribution, difficulty_gate, risk_gate

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

    def _compute_alpha_eff(self, risk_gate: torch.Tensor):
        risk_batch = self._summarize_risk(risk_gate)
        lambda_ret_batch = self.lambda_ret_lo + (
            self.lambda_ret_hi - self.lambda_ret_lo
        ) * risk_batch
        alpha_eff = self.alpha * lambda_ret_batch
        return alpha_eff, lambda_ret_batch, risk_batch

    def _build_neg_mask(self, original_inputs, alternate_inputs):
        if self.local_neg_mode == "full":
            return None
        if self.local_neg_mode == "answer_only":
            return build_answer_only_mask(original_inputs)
        if self.local_neg_mode == "answer_minus_shared":
            return build_answer_minus_shared_mask(original_inputs, alternate_inputs)
        raise ValueError(f"Unknown local_neg_mode={self.local_neg_mode}")

    def _effective_belief_beta(self) -> float:
        return self.beta if self.belief_beta is None else self.belief_beta

    def _compute_forget_loss_vec(self, model, forget_inputs):
        original_inputs = forget_inputs["original"]
        alternate_inputs = forget_inputs["alternate"]

        cf_vec, alternate_outputs = compute_nll_per_sample(
            model,
            alternate_inputs,
            normalize_by_tokens=self.normalize_cf_by_tokens,
        )

        batch_size = int(cf_vec.shape[0])
        device = cf_vec.device
        difficulty, attribution, difficulty_gate, risk_gate = self._routing_state(
            forget_inputs=forget_inputs,
            device=device,
            batch_size=batch_size,
        )

        neg_mask = self._build_neg_mask(original_inputs, alternate_inputs)
        if neg_mask is None:
            neg_vec, _ = compute_npo_per_sample(
                model,
                self.ref_model,
                original_inputs,
                beta=self.beta,
                normalize_by_tokens=self.normalize_neg_by_tokens,
            )
            neg_mask_frac = torch.tensor(1.0, device=device)
        else:
            neg_vec, _ = compute_npo_per_sample_masked(
                model,
                self.ref_model,
                original_inputs,
                beta=self.beta,
                token_mask=neg_mask,
                normalize_by_tokens=self.normalize_neg_by_tokens,
            )
            neg_mask_frac = neg_mask.float().mean().detach()

        lambda_neg = self.lambda_neg_max * difficulty_gate * (1.0 - risk_gate)
        forget_scale = 1.0 - (1.0 - self.risk_forget_scale) * risk_gate

        belief_neg_vec = torch.zeros_like(neg_vec)
        lambda_belief = torch.zeros_like(lambda_neg)
        belief_inputs = forget_inputs.get("belief")
        if belief_inputs is not None and self.belief_neg_weight > 0.0:
            belief_neg_vec, _ = compute_npo_per_sample(
                model,
                self.ref_model,
                belief_inputs,
                beta=self._effective_belief_beta(),
                normalize_by_tokens=self.normalize_neg_by_tokens,
            )
            lambda_belief = self.belief_neg_weight * difficulty_gate * (
                1.0 - 0.5 * risk_gate
            )

        forget_vec = forget_scale * (
            self.cf_weight * cf_vec
            + lambda_neg * neg_vec
            + lambda_belief * belief_neg_vec
        )

        aux = {
            "cf_vec": cf_vec,
            "neg_vec": neg_vec,
            "belief_neg_vec": belief_neg_vec,
            "difficulty": difficulty,
            "attribution": attribution,
            "difficulty_gate": difficulty_gate,
            "risk_gate": risk_gate,
            "lambda_neg": lambda_neg,
            "lambda_belief": lambda_belief,
            "forget_scale": forget_scale,
            "neg_mask_frac": neg_mask_frac,
        }
        return forget_vec, alternate_outputs, aux

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_vec, alternate_outputs, aux = self._compute_forget_loss_vec(
            model=model,
            forget_inputs=inputs["forget"],
        )

        alpha_eff, lambda_ret_batch, risk_batch = self._compute_alpha_eff(
            aux["risk_gate"]
        )
        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )
        forget_loss = forget_vec.mean()
        loss = self.gamma * forget_loss + alpha_eff * retain_loss

        try:
            alpha_eff_value = (
                float(alpha_eff.detach().item())
                if torch.is_tensor(alpha_eff)
                else float(alpha_eff)
            )
            self.log(
                {
                    "dualcf_cf_loss": float(aux["cf_vec"].mean().detach().item()),
                    "dualcf_neg_loss": float(aux["neg_vec"].mean().detach().item()),
                    "dualcf_belief_neg_loss": float(
                        aux["belief_neg_vec"].mean().detach().item()
                    ),
                    "dualcf_forget_loss": float(forget_loss.detach().item()),
                    "dualcf_retain_loss": float(retain_loss.detach().item()),
                    "dualcf_alpha_eff": alpha_eff_value,
                    "dualcf_lambda_ret_batch": float(
                        lambda_ret_batch.detach().item()
                    ),
                    "dualcf_d_mean": float(aux["difficulty"].mean().detach().item()),
                    "dualcf_a_mean": float(aux["attribution"].mean().detach().item()),
                    "dualcf_s_mean": float(
                        aux["difficulty_gate"].mean().detach().item()
                    ),
                    "dualcf_s_p50": float(
                        torch.quantile(aux["difficulty_gate"], 0.50).detach().item()
                    ),
                    "dualcf_s_p90": float(
                        torch.quantile(aux["difficulty_gate"], 0.90).detach().item()
                    ),
                    "dualcf_r_mean": float(aux["risk_gate"].mean().detach().item()),
                    "dualcf_r_p50": float(
                        torch.quantile(aux["risk_gate"], 0.50).detach().item()
                    ),
                    "dualcf_r_p90": float(
                        torch.quantile(aux["risk_gate"], 0.90).detach().item()
                    ),
                    "dualcf_r_hi_frac": float(
                        (aux["risk_gate"] > 0.8).float().mean().detach().item()
                    ),
                    "dualcf_risk_batch": float(risk_batch.detach().item()),
                    "dualcf_lambda_neg_mean": float(
                        aux["lambda_neg"].mean().detach().item()
                    ),
                    "dualcf_lambda_belief_mean": float(
                        aux["lambda_belief"].mean().detach().item()
                    ),
                    "dualcf_neg_mask_frac": float(
                        aux["neg_mask_frac"].detach().item()
                    ),
                }
            )
        except Exception:
            pass

        return (loss, alternate_outputs) if return_outputs else loss
