import math
import logging
from typing import Dict, Any, Optional

import torch

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    compute_wga_loss_dynamic_beta,
    beta_from_pop_sum_tensor,
)


class AdaWGD(GradDiff):
    """
    Adaptive WGA-based unlearning with dynamic popularity weighting, optional anti-repetition,
    an adaptive retain constraint, and optional forget warmup.

    Total loss per step: L_total = gamma_k * L_forget + alpha_k * L_retain

    - L_forget: WGA with beta from pop_sum and optional anti-repetition penalty
    - L_retain: NLL or KL(current||ref) on retain
    - gamma_k: linear warmup over Kw epochs to gamma_final
    - alpha_k: alpha0 + lambda_k, where lambda_k is adapted at each epoch-end based on retain degradation
    """

    def __init__(
        self,
        # Forget-side params
        gamma: float = 1.0,  # final gamma
        rep_coeff: float = 0.0,
        warmup_epochs: float = 0.0,  # Kw; 0 disables warmup
        # Retain constraint params (epoch-wise controller, no small-batch evals)
        alpha0: float = 0.5,  # starting alpha; adapted within [0, gamma]
        alpha_const: Optional[float] = None,  # if set, use constant alpha instead of adapting
        # Forget-side beta control
        beta_const: Optional[float] = None,   # if set, use constant per-sample beta instead of dynamic from pop_sum
        eps: float = 0.1,  # target relative tolerance on retain drift
        tau: float = 0.05,  # dead-zone half-width around eps
        # Online EMA smoothing for signals
        retain_ema_decay: float = 0.9,
        forget_ema_decay: float = 0.9,
        # Per-epoch adjustment magnitudes
        eta_big: float = 0.1,
        eta_small: float = 0.05,
        # Sensitivities
        dF_tol: float = 0.02,  # |relative forget change| below this is "flat"
        k_dist: float = 1.0,   # scale factor for distance beyond tolerance
        k_flat: float = 1.0,   # extra scale when both signals are flat
        mcap: float = 3.0,     # cap on distance multiplier
        # Popularity modulation of Δα (epoch-wise) — can be disabled or tuned
        pop_delta_alpha_enable: bool = True,
        pop_inc_amp: float = 0.5,   # in [0,1]; s_inc = (1 - pop_inc_amp) + pop_inc_amp * z
        pop_dec_amp: float = 0.5,   # in [0,1]; s_dec = 1 - pop_dec_amp * z
        # Step logging interval for richer training logs (in steps)
        step_log_interval: int = 50,
        # Popularity-coupled retain scaling (optional)
        retain_pop_mode: str = "none",  # one of: "none", "inv_beta"
        retain_pop_gain: float = 1.0,   # strength for scaling with mean beta
        retain_pop_min: float = 0.3,    # lower clamp for scaling factor
        retain_pop_max: float = 1.0,    # upper clamp for scaling factor
        *args,
        **kwargs,
    ):
        # Be compatible with WGA experiment presets that pass an unused 'beta'
        if "beta" in kwargs:
            kwargs.pop("beta", None)
        super().__init__(*args, **kwargs)
        # Save method params
        self.gamma_final = gamma
        self.rep_coeff = rep_coeff
        self.warmup_epochs = float(warmup_epochs)

        self.alpha0 = float(alpha0)
        self.alpha_const = float(alpha_const) if (alpha_const is not None) else None
        self.beta_const = float(beta_const) if (beta_const is not None) else None
        self.eps = float(eps)
        self.tau = float(tau)
        self.retain_ema_decay = float(retain_ema_decay)
        self.forget_ema_decay = float(forget_ema_decay)
        self.eta_big = float(eta_big)
        self.eta_small = float(eta_small)
        self.dF_tol = float(dF_tol)
        self.k_dist = float(k_dist)
        self.k_flat = float(k_flat)
        self.mcap = float(mcap)
        self.pop_delta_alpha_enable = bool(pop_delta_alpha_enable)
        self.pop_inc_amp = float(pop_inc_amp)
        self.pop_dec_amp = float(pop_dec_amp)
        self.step_log_interval = int(step_log_interval)
        # popularity-coupled retain scaling params
        self.retain_pop_mode = str(retain_pop_mode).lower()
        self.retain_pop_gain = float(retain_pop_gain)
        self.retain_pop_min = float(retain_pop_min)
        self.retain_pop_max = float(retain_pop_max)

        # State variables
        self.lambda_k = 0.0
        self.alpha_k = self.alpha0
        self.gamma_k = 0.0 if self.warmup_epochs > 0 else self.gamma_final

        # Ensure ref model exists if retain uses KL in base logic
        if self.ref_model is None:
            try:
                self.ref_model = self._prepare_ref_model(self.model)
            except Exception:
                pass

        # Online EMAs (initialized lazily during training)
        self._retain_ema: Optional[float] = None
        self._forget_strength_ema: Optional[float] = None
        self._prev_forget_strength_ema: Optional[float] = None
        self._beta_mean_ema: Optional[float] = None
        self._ret_baseline: Optional[float] = None
        # Attach epoch-end adaptation callback
        try:
            self.add_callback(AdaWGDCallback(self))
        except Exception:
            pass
        self._logger = logging.getLogger(__name__)

    # ------------------------ Core Training Loss ------------------------
    def _current_epoch(self) -> float:
        try:
            return float(getattr(self.state, "epoch", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _update_gamma_schedule(self):
        if self.warmup_epochs > 0:
            k = max(0.0, self._current_epoch())
            self.gamma_k = self.gamma_final * min(1.0, k / self.warmup_epochs)
        else:
            self.gamma_k = self.gamma_final

    def compute_loss(self, model, inputs, return_outputs=False):
        # Update schedules
        self._update_gamma_schedule()
        # Persisted alpha; clamp within [0, gamma_k]
        if self.alpha_const is not None:
            # Constant-alpha mode (user-specified); ignore lambda adaptation
            self.alpha_k = float(max(0.0, min(self.gamma_k, self.alpha_const)))
            # keep lambda_k consistent for logging only
            self.lambda_k = self.alpha_k - self.alpha0
        else:
            self.alpha_k = float(max(0.0, min(self.gamma_k, self.alpha0 + self.lambda_k)))

        # Forget inputs
        finputs_full = inputs["forget"]
        forget_inputs = {k: finputs_full[k] for k in ("input_ids", "attention_mask", "labels", "pop_sum") if k in finputs_full}
        # Training-time repetition penalty disabled for AdaWGD; handle repetition at inference
        forget_loss, forget_outputs = compute_wga_loss_dynamic_beta(
            model=model,
            inputs=forget_inputs,
            beta_from_pop_sum=True,
            rep_coeff=0.0,
            beta_const=self.beta_const,
        )

        # Retain inputs
        rinputs_full = inputs["retain"]
        retain_inputs = {
            "input_ids": rinputs_full["input_ids"],
            "attention_mask": rinputs_full["attention_mask"],
            "labels": rinputs_full["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # Popularity-coupled retain scaling (based on current forget batch)
        alpha_eff = self.alpha_k
        pop_beta_mean = None
        # Always compute mean beta if pop_sum is available (for logging/EMA),
        # but only apply alpha scaling when retain_pop_mode is enabled.
        finputs_for_pop = inputs["forget"]
        if "pop_sum" in finputs_for_pop:
            pop_sum = finputs_for_pop["pop_sum"].to(self.accelerator.device).float().view(-1)
            beta_vec = beta_from_pop_sum_tensor(pop_sum)
            pop_beta_mean = float(beta_vec.mean().detach().item())
            if self.retain_pop_mode == "inv_beta":
                # scale alpha inversely with mean beta: more popular => lower retain penalty
                scale = 1.0 / (1.0 + self.retain_pop_gain * max(0.0, pop_beta_mean))
                scale = float(max(self.retain_pop_min, min(self.retain_pop_max, scale)))
                alpha_eff = self.alpha_k * scale

        loss = self.gamma_k * forget_loss + alpha_eff * retain_loss

        # Stash last-step effective alpha and beta mean for epoch logs
        self._last_alpha_eff = float(alpha_eff)
        self._last_beta_mean = float(pop_beta_mean) if pop_beta_mean is not None else None

        # Light logging of effective alpha and batch popularity when enabled
        if self.retain_pop_mode != "none" and pop_beta_mean is not None:
            self.log({
                "ada_alpha_eff": float(alpha_eff),
                "ada_beta_mean_forget": float(pop_beta_mean) if pop_beta_mean is not None else 0.0,
            })
        
        # Rich step-level logging at a configurable interval to the Trainer logs.
        try:
            self._step_count = getattr(self, "_step_count", 0) + 1
            self._last_logged_step = getattr(self, "_last_logged_step", 0)
            if self.step_log_interval > 0 and (self._step_count - self._last_logged_step) >= self.step_log_interval:
                self._last_logged_step = self._step_count
                step_payload = {
                    "ada_step_forget_loss": float(forget_loss.detach().item()),
                    "ada_step_retain_loss": float(retain_loss.detach().item()),
                    "ada_step_total_loss": float(loss.detach().item()),
                    "ada_step_alpha_k": float(self.alpha_k),
                    "ada_step_alpha_eff": float(alpha_eff),
                    "ada_step_gamma_k": float(self.gamma_k),
                    "ada_step_beta_mean": float(pop_beta_mean) if pop_beta_mean is not None else 0.0,
                    "ada_step_beta_mean_ema": float(self._beta_mean_ema) if getattr(self, "_beta_mean_ema", None) is not None else 0.0,
                    "ada_step_ret_ema": float(self._retain_ema) if getattr(self, "_retain_ema", None) is not None else 0.0,
                    "ada_step_forget_strength_ema": float(self._forget_strength_ema) if getattr(self, "_forget_strength_ema", None) is not None else 0.0,
                }
                self.log(step_payload)
                # Extra stdout log so AdaWGD.log captures step-wise summaries.
                self._logger.info(
                    "[AdaWGD][step %s] total=%.4f forget=%.4f retain=%.4f alpha=%.4f alpha_eff=%.4f gamma=%.4f beta_mean=%.4f beta_ema=%.4f",
                    self._step_count,
                    step_payload["ada_step_total_loss"],
                    step_payload["ada_step_forget_loss"],
                    step_payload["ada_step_retain_loss"],
                    step_payload["ada_step_alpha_k"],
                    step_payload["ada_step_alpha_eff"],
                    step_payload["ada_step_gamma_k"],
                    step_payload["ada_step_beta_mean"],
                    step_payload["ada_step_beta_mean_ema"],
                )
                try:
                    self.accelerator.print(
                        f"[AdaWGD][step {self._step_count}] total={step_payload['ada_step_total_loss']:.4f} "
                        f"forget={step_payload['ada_step_forget_loss']:.4f} retain={step_payload['ada_step_retain_loss']:.4f} "
                        f"alpha={step_payload['ada_step_alpha_k']:.4f} alpha_eff={step_payload['ada_step_alpha_eff']:.4f} "
                        f"gamma={step_payload['ada_step_gamma_k']:.4f} beta_mean={step_payload['ada_step_beta_mean']:.4f} "
                        f"beta_ema={step_payload['ada_step_beta_mean_ema']:.4f}"
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Update online EMAs
        try:
            r_val = float(retain_loss.detach().item())
            f_strength = float(-forget_loss.detach().item())  # positive when forgetting is stronger
            if self._retain_ema is None:
                self._retain_ema = r_val
            else:
                self._retain_ema = self.retain_ema_decay * self._retain_ema + (1.0 - self.retain_ema_decay) * r_val
            if self._forget_strength_ema is None:
                self._forget_strength_ema = f_strength
            else:
                self._forget_strength_ema = self.forget_ema_decay * self._forget_strength_ema + (1.0 - self.forget_ema_decay) * f_strength
            if pop_beta_mean is not None:
                if self._beta_mean_ema is None:
                    self._beta_mean_ema = pop_beta_mean
                else:
                    self._beta_mean_ema = 0.9 * self._beta_mean_ema + 0.1 * pop_beta_mean
        except Exception:
            pass
        return (loss, forget_outputs) if return_outputs else loss

    # ------------------------ Epoch-end Adaptation ------------------------
    @torch.no_grad()
    def post_epoch_update(self):
        # If constant-alpha mode is enabled, keep alpha fixed and only log retain/forget signals
        if self.alpha_const is not None:
            Rk = float(self._retain_ema) if self._retain_ema is not None else 0.0
            if not hasattr(self, "_ret_baseline") or self._ret_baseline is None:
                self._ret_baseline = Rk
            delta_rel = max(0.0, (Rk - (self._ret_baseline or 1e-8)) / max((self._ret_baseline or 1e-8), 1e-8)) if (self._ret_baseline or 0.0) > 0 else 0.0
            self.log({
                "ada_lambda": float(self.lambda_k),
                "ada_alpha_k": float(self.alpha_k),
                "ada_gamma_k": float(self.gamma_k),
                "ada_delta_k": float(delta_rel),
                "ada_ret_ema": float(Rk),
                "ada_ret_ref": float(self._ret_baseline or 0.0),
                "ada_dF": float(0.0),
                "ada_alpha_delta": float(0.0),
            })
            self._logger.info(
                "[AdaWGD][epoch end] alpha=%.4f lambda=%.4f gamma=%.4f delta=%.4f ret_ema=%.4f ret_ref=%.4f beta_ema=%.4f",
                float(self.alpha_k),
                float(self.lambda_k),
                float(self.gamma_k),
                float(delta_rel),
                float(Rk),
                float(self._ret_baseline or 0.0),
                float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0,
            )
            try:
                self.accelerator.print(
                    f"[AdaWGD][epoch end] alpha={float(self.alpha_k):.4f} lambda={float(self.lambda_k):.4f} "
                    f"gamma={float(self.gamma_k):.4f} delta={float(delta_rel):.4f} ret_ema={float(Rk):.4f} "
                    f"ret_ref={float(self._ret_baseline or 0.0):.4f} "
                    f"beta_ema={float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0:.4f}"
                )
            except Exception:
                pass
            return

        # Establish baseline retain reference lazily on first epoch end
        Rk = float(self._retain_ema) if self._retain_ema is not None else None
        Fk = float(self._forget_strength_ema) if self._forget_strength_ema is not None else None

        if self._ret_baseline is None and Rk is not None:
            self._ret_baseline = Rk

        # Relative retain drift (one-sided)
        if Rk is not None and self._ret_baseline is not None and self._ret_baseline > 0:
            delta_rel = max(0.0, (Rk - self._ret_baseline) / max(self._ret_baseline, 1e-8))
        else:
            delta_rel = 0.0

        # Relative change in forget strength across epochs
        if Fk is not None and self._prev_forget_strength_ema is not None:
            denom = max(abs(self._prev_forget_strength_ema), 1e-6)
            dF = (Fk - self._prev_forget_strength_ema) / denom
        else:
            dF = 0.0
        if Fk is not None:
            self._prev_forget_strength_ema = Fk

        # Decide alpha update magnitude
        T = self.eps
        far = delta_rel - (T + self.tau)
        near = delta_rel - T
        if far > 0:
            m_dist = min(self.mcap, max(0.0, far / max(T, 1e-6)))
            delta_alpha = self.eta_big * (1.0 + self.k_dist * m_dist)
        elif abs(near) <= self.tau and abs(dF) <= self.dF_tol:
            delta_alpha = -self.eta_big * (1.0 + self.k_flat)
        else:
            if near > 0:
                delta_alpha = self.eta_small
            elif near < 0 and dF > self.dF_tol:
                delta_alpha = -self.eta_small
            else:
                delta_alpha = 0.0

        # Popularity-scaled magnitude based on EMA of mean beta (optional)
        if self.pop_delta_alpha_enable and self._beta_mean_ema is not None:
            z = (self._beta_mean_ema - 0.05) / max(1e-6, (2.0 - 0.05))
            z = float(max(0.0, min(1.0, z)))
            if delta_alpha > 0:
                s_inc = (1.0 - self.pop_inc_amp) + self.pop_inc_amp * z
                delta_alpha *= s_inc
            elif delta_alpha < 0:
                s_dec = 1.0 - self.pop_dec_amp * z
                delta_alpha *= s_dec

        # Apply and clamp within [0, gamma_k]
        cur_alpha = float(max(0.0, min(self.gamma_k, self.alpha0 + self.lambda_k)))
        new_alpha = max(0.0, min(self.gamma_k, cur_alpha + delta_alpha))
        self.lambda_k = new_alpha - self.alpha0
        self.alpha_k = new_alpha

        # Log adaptation state
        self.log(
            {
                "ada_lambda": float(self.lambda_k),
                "ada_alpha_k": float(self.alpha_k),
                "ada_gamma_k": float(self.gamma_k),
                "ada_delta_k": float(delta_rel),
                "ada_ret_ema": float(Rk) if Rk is not None else 0.0,
                "ada_ret_ref": float(self._ret_baseline) if self._ret_baseline is not None else 0.0,
                "ada_dF": float(dF),
                "ada_alpha_delta": float(delta_alpha),
                "ada_beta_mean": float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0,
                # Final/current retain alpha used (epoch-level view) and last-step alpha_eff snapshot
                "ada_alpha_final": float(self.alpha_k),
                "ada_alpha_eff_last": float(getattr(self, "_last_alpha_eff", self.alpha_k)),
            }
        )
        # Epoch-end stdout log for AdaWGD.log
        self._logger.info(
            "[AdaWGD][epoch end] alpha=%.4f lambda=%.4f gamma=%.4f delta=%.4f dF=%.4f ret_ema=%.4f ret_ref=%.4f beta_ema=%.4f",
            float(self.alpha_k),
            float(self.lambda_k),
            float(self.gamma_k),
            float(delta_rel),
            float(dF),
            float(Rk) if Rk is not None else 0.0,
            float(self._ret_baseline) if self._ret_baseline is not None else 0.0,
            float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0,
        )
        try:
            self.accelerator.print(
                f"[AdaWGD][epoch end] alpha={float(self.alpha_k):.4f} lambda={float(self.lambda_k):.4f} "
                f"gamma={float(self.gamma_k):.4f} delta={float(delta_rel):.4f} dF={float(dF):.4f} "
                f"ret_ema={float(Rk) if Rk is not None else 0.0:.4f} "
                f"ret_ref={float(self._ret_baseline) if self._ret_baseline is not None else 0.0:.4f} "
                f"beta_ema={float(self._beta_mean_ema) if self._beta_mean_ema is not None else 0.0:.4f}"
            )
        except Exception:
            pass

    # Hook into epoch end via callback-like method
    def _maybe_post_epoch(self):
        # Called by callback at epoch end
        self.post_epoch_update()


from transformers import TrainerCallback


class AdaWGDCallback(TrainerCallback):
    def __init__(self, trainer: AdaWGD):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Expose control back to trainer for early stop
        self.trainer.control = control
        self.trainer._maybe_post_epoch()
