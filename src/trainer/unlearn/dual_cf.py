import math

import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_nll_per_sample, compute_npo_per_sample


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

        if self.beta <= 0.0:
            raise ValueError("DualCF requires beta > 0.")
        if self.temp_d <= 0.0 or self.temp_a <= 0.0:
            raise ValueError("DualCF requires temp_d > 0 and temp_a > 0.")
        if self.alpha_eff_topk_frac <= 0.0 or self.alpha_eff_topk_frac > 1.0:
            raise ValueError("DualCF requires 0 < alpha_eff_topk_frac <= 1.")

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

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        original_inputs = forget_inputs["original"]
        alternate_inputs = forget_inputs["alternate"]

        cf_vec, alternate_outputs = compute_nll_per_sample(
            model,
            alternate_inputs,
            normalize_by_tokens=self.normalize_cf_by_tokens,
        )
        neg_vec, _ = compute_npo_per_sample(
            model,
            self.ref_model,
            original_inputs,
            beta=self.beta,
            normalize_by_tokens=self.normalize_neg_by_tokens,
        )

        batch_size = int(cf_vec.shape[0])
        device = cf_vec.device
        difficulty = self._score_tensor(
            forget_inputs, "difficulty_score", device=device, batch_size=batch_size
        )
        attribution = self._score_tensor(
            forget_inputs, "attribution_score", device=device, batch_size=batch_size
        )

        if self.disable_difficulty_route:
            difficulty = torch.zeros_like(difficulty)
        if self.disable_attribution_route:
            attribution = torch.zeros_like(attribution)

        difficulty_gate = torch.sigmoid((difficulty - self.tau_d) / self.temp_d)
        risk_gate = torch.sigmoid((attribution - self.tau_a) / self.temp_a)
        difficulty_gate = difficulty_gate.pow(self.neg_power)
        risk_gate = risk_gate.pow(self.risk_power)

        lambda_neg = self.lambda_neg_max * difficulty_gate * (1.0 - risk_gate)
        forget_scale = 1.0 - (1.0 - self.risk_forget_scale) * risk_gate

        forget_loss = (
            forget_scale * (self.cf_weight * cf_vec + lambda_neg * neg_vec)
        ).mean()

        risk_batch = self._summarize_risk(risk_gate)
        lambda_ret_batch = self.lambda_ret_lo + (
            self.lambda_ret_hi - self.lambda_ret_lo
        ) * risk_batch
        alpha_eff = self.alpha * lambda_ret_batch

        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )
        loss = self.gamma * forget_loss + alpha_eff * retain_loss

        try:
            alpha_eff_value = (
                float(alpha_eff.detach().item())
                if torch.is_tensor(alpha_eff)
                else float(alpha_eff)
            )
            self.log(
                {
                    "dualcf_cf_loss": float(cf_vec.mean().detach().item()),
                    "dualcf_neg_loss": float(neg_vec.mean().detach().item()),
                    "dualcf_forget_loss": float(forget_loss.detach().item()),
                    "dualcf_retain_loss": float(retain_loss.detach().item()),
                    "dualcf_alpha_eff": alpha_eff_value,
                    "dualcf_lambda_ret_batch": float(
                        lambda_ret_batch.detach().item()
                    ),
                    "dualcf_d_mean": float(difficulty.mean().detach().item()),
                    "dualcf_a_mean": float(attribution.mean().detach().item()),
                    "dualcf_s_mean": float(difficulty_gate.mean().detach().item()),
                    "dualcf_s_p50": float(
                        torch.quantile(difficulty_gate, 0.50).detach().item()
                    ),
                    "dualcf_s_p90": float(
                        torch.quantile(difficulty_gate, 0.90).detach().item()
                    ),
                    "dualcf_r_mean": float(risk_gate.mean().detach().item()),
                    "dualcf_r_p50": float(
                        torch.quantile(risk_gate, 0.50).detach().item()
                    ),
                    "dualcf_r_p90": float(
                        torch.quantile(risk_gate, 0.90).detach().item()
                    ),
                    "dualcf_r_hi_frac": float(
                        (risk_gate > 0.8).float().mean().detach().item()
                    ),
                    "dualcf_risk_batch": float(risk_batch.detach().item()),
                    "dualcf_lambda_neg_mean": float(
                        lambda_neg.mean().detach().item()
                    ),
                }
            )
        except Exception:
            pass

        return (loss, alternate_outputs) if return_outputs else loss
