from typing import Optional

import torch

from transformers import TrainerCallback
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss_dynamic_beta, beta_from_pop_sum_tensor


class AdaPop(GradDiff):
    """
    Adaptive WGA with a principled dual-ascent retain constraint.

    Constraint form: minimize L_f subject to Δ_r <= eps.
    Lagrangian: L = gamma * L_f + alpha * L_r, alpha = alpha0 + lambda
    Dual update: lambda <- clip(lambda + dual_lr * (Δ_r - eps), [0, lambda_max])
    """

    def __init__(
        self,
        gamma: float = 1.0,
        warmup_epochs: float = 0.0,
        alpha0: float = 0.5,
        alpha_const: Optional[float] = None,
        beta_const: Optional[float] = None,
        eps: float = 0.1,
        dual_lr: float = 0.1,
        lambda_max: float = 5.0,
        retain_ema_decay: float = 0.9,
        step_log_interval: int = 50,
        retain_pop_mode: str = "none",
        retain_pop_gain: float = 1.0,
        retain_pop_min: float = 0.3,
        retain_pop_max: float = 1.0,
        *args,
        **kwargs,
    ):
        if "beta" in kwargs:
            kwargs.pop("beta", None)
        super().__init__(*args, **kwargs)

        self.gamma_final = float(gamma)
        self.warmup_epochs = float(warmup_epochs)
        self.alpha0 = float(alpha0)
        self.alpha_const = float(alpha_const) if alpha_const is not None else None
        self.beta_const = float(beta_const) if beta_const is not None else None

        self.eps = float(eps)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)
        self.retain_ema_decay = float(retain_ema_decay)
        self.step_log_interval = int(step_log_interval)

        self.retain_pop_mode = str(retain_pop_mode).lower()
        self.retain_pop_gain = float(retain_pop_gain)
        self.retain_pop_min = float(retain_pop_min)
        self.retain_pop_max = float(retain_pop_max)

        self.lambda_k = 0.0
        self.alpha_k = self.alpha0
        self.gamma_k = 0.0 if self.warmup_epochs > 0 else self.gamma_final

        self._retain_ema: Optional[float] = None
        self._ret_baseline: Optional[float] = None
        self._beta_mean_ema: Optional[float] = None
        self._last_alpha_eff: float = 0.0
        self._last_beta_mean: Optional[float] = None

        try:
            self.add_callback(AdaPopCallback(self))
        except Exception:
            pass

    def _current_epoch(self) -> float:
        try:
            return float(getattr(self.state, "epoch", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _update_gamma_schedule(self) -> None:
        if self.warmup_epochs > 0:
            k = max(0.0, self._current_epoch())
            self.gamma_k = self.gamma_final * min(1.0, k / self.warmup_epochs)
        else:
            self.gamma_k = self.gamma_final

    def _update_retain_ema(self, retain_loss: torch.Tensor) -> float:
        value = float(retain_loss.detach().item())
        if self._retain_ema is None:
            self._retain_ema = value
        else:
            d = self.retain_ema_decay
            self._retain_ema = d * self._retain_ema + (1.0 - d) * value
        return float(self._retain_ema)

    def compute_loss(self, model, inputs, return_outputs=False):
        self._update_gamma_schedule()

        if self.alpha_const is not None:
            self.alpha_k = float(max(0.0, min(self.gamma_k, self.alpha_const)))
            self.lambda_k = self.alpha_k - self.alpha0
        else:
            self.alpha_k = float(max(0.0, min(self.gamma_k, self.alpha0 + self.lambda_k)))

        finputs_full = inputs["forget"]
        forget_inputs = {
            k: finputs_full[k]
            for k in ("input_ids", "attention_mask", "labels", "pop_sum")
            if k in finputs_full
        }
        forget_loss, forget_outputs = compute_wga_loss_dynamic_beta(
            model=model,
            inputs=forget_inputs,
            beta_from_pop_sum=True,
            rep_coeff=0.0,
            beta_const=self.beta_const,
        )

        rinputs_full = inputs["retain"]
        retain_inputs = {
            "input_ids": rinputs_full["input_ids"],
            "attention_mask": rinputs_full["attention_mask"],
            "labels": rinputs_full["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        retain_ema = self._update_retain_ema(retain_loss)

        alpha_eff = self.alpha_k
        pop_beta_mean = None
        if "pop_sum" in finputs_full:
            pop_sum = finputs_full["pop_sum"].to(self.accelerator.device).float().view(-1)
            beta_vec = beta_from_pop_sum_tensor(pop_sum)
            pop_beta_mean = float(beta_vec.mean().detach().item())
            if self._beta_mean_ema is None:
                self._beta_mean_ema = pop_beta_mean
            else:
                self._beta_mean_ema = 0.9 * self._beta_mean_ema + 0.1 * pop_beta_mean
            if self.retain_pop_mode == "inv_beta":
                scale = 1.0 / (1.0 + self.retain_pop_gain * max(0.0, pop_beta_mean))
                scale = float(max(self.retain_pop_min, min(self.retain_pop_max, scale)))
                alpha_eff = self.alpha_k * scale

        loss = self.gamma_k * forget_loss + alpha_eff * retain_loss

        self._last_alpha_eff = float(alpha_eff)
        self._last_beta_mean = float(pop_beta_mean) if pop_beta_mean is not None else None

        try:
            self._step_count = getattr(self, "_step_count", 0) + 1
            self._last_logged_step = getattr(self, "_last_logged_step", 0)
            if self.step_log_interval > 0 and (self._step_count - self._last_logged_step) >= self.step_log_interval:
                self._last_logged_step = self._step_count
                self.log(
                    {
                        "adapop_step_forget_loss": float(forget_loss.detach().item()),
                        "adapop_step_retain_loss": float(retain_loss.detach().item()),
                        "adapop_step_alpha_k": float(self.alpha_k),
                        "adapop_step_alpha_eff": float(alpha_eff),
                        "adapop_step_gamma_k": float(self.gamma_k),
                        "adapop_step_ret_ema": float(retain_ema),
                        "adapop_step_beta_mean": float(pop_beta_mean) if pop_beta_mean is not None else 0.0,
                    }
                )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss

    @torch.no_grad()
    def post_epoch_update(self) -> None:
        if self._retain_ema is None:
            return
        if self._ret_baseline is None:
            self._ret_baseline = float(self._retain_ema)
            return

        base = max(self._ret_baseline, 1e-8)
        delta = max(0.0, (float(self._retain_ema) - self._ret_baseline) / base)

        if self.alpha_const is None:
            self.lambda_k = float(
                max(0.0, min(self.lambda_max, self.lambda_k + self.dual_lr * (delta - self.eps)))
            )
            self.alpha_k = float(max(0.0, min(self.gamma_k, self.alpha0 + self.lambda_k)))

        self.log(
            {
                "adapop_lambda": float(self.lambda_k),
                "adapop_alpha_k": float(self.alpha_k),
                "adapop_gamma_k": float(self.gamma_k),
                "adapop_delta": float(delta),
                "adapop_ret_ema": float(self._retain_ema),
                "adapop_ret_ref": float(self._ret_baseline),
                "adapop_beta_mean": float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0,
                "adapop_alpha_eff_last": float(self._last_alpha_eff),
            }
        )


class AdaPopCallback(TrainerCallback):
    def __init__(self, trainer: AdaPop):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            self.trainer.post_epoch_update()
        except Exception:
            pass
        return control
