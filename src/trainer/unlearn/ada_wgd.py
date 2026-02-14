from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    compute_wga_loss_dynamic_beta,
    compute_kl_divergence,
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
        # Retain constraint params
        alpha0: float = 0.5,  # base alpha
        eps: float = 0.1,  # relative tolerance on retain degradation (e.g., 0.1 = 10%)
        tau: float = 0.05,  # dead-zone band (relative)
        delta: float = 0.1,  # legacy gain; use controller_gain when provided
        controller_gain: Optional[float] = None,  # preferred: sets kp; ki = kp/2
        lambda_max: float = 1000.0,
        # Small eval subsets
        retain_eval_samples: int = 128,
        forget_eval_samples: int = 128,
        # Early stopping thresholds
        forget_improve_eps: float = 0.005,
        consecutive_patience: int = 2,
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
        # Constraint tolerance treated as RELATIVE tolerance (fraction of L_ret_ref)
        # i.e., delta_rel = (L_ret_cur - L_ret_ref) / max(L_ret_ref, 1e-8)
        # and we desire delta_rel <= eps within a deadband of +/- tau
        self.eps = float(eps)
        self.tau = float(tau)

        # Controller gains (prefer controller_gain; else map legacy 'delta')
        if controller_gain is not None:
            self.kp = float(controller_gain)
            self.ki = float(controller_gain) / 2.0
        else:
            self.kp = float(delta)
            self.ki = float(delta) / 2.0
        self.lambda_max = float(lambda_max)

        self.retain_eval_samples = int(retain_eval_samples)
        self.forget_eval_samples = int(forget_eval_samples)
        self.forget_improve_eps = float(forget_improve_eps)
        self.consecutive_patience = int(consecutive_patience)
        # popularity-coupled retain scaling params
        self.retain_pop_mode = str(retain_pop_mode).lower()
        self.retain_pop_gain = float(retain_pop_gain)
        self.retain_pop_min = float(retain_pop_min)
        self.retain_pop_max = float(retain_pop_max)

        # State variables
        self.lambda_k = 0.0
        self.alpha_k = self.alpha0
        self.gamma_k = 0.0 if self.warmup_epochs > 0 else self.gamma_final

        # Make sure reference model exists if needed (KL or baseline retain eval)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        # Build small retain/forget subsets for quick epoch-end measurements
        self._prepare_eval_subsets()

        # Precompute baseline retain loss on reference model for NLL; for KL we use dynamic KL
        self.L_ret_ref = self._compute_retain_loss(self.ref_model)

        # Early stopping trackers
        self.prev_forget_loss = None
        self.prev_bad_epochs = 0
        # PI integrator state for retain constraint; anti-windup via clamping
        self._pi_err_int = 0.0
        # Attach epoch-end adaptation callback
        try:
            self.add_callback(AdaWGDCallback(self))
        except Exception:
            pass

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
        self.alpha_k = self.alpha0 + self.lambda_k

        # Forget inputs
        finputs_full = inputs["forget"]
        forget_inputs = {k: finputs_full[k] for k in ("input_ids", "attention_mask", "labels", "pop_sum") if k in finputs_full}
        # Training-time repetition penalty disabled for AdaWGD; handle repetition at inference
        forget_loss, forget_outputs = compute_wga_loss_dynamic_beta(
            model=model, inputs=forget_inputs, beta_from_pop_sum=True, rep_coeff=0.0
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
        if self.retain_pop_mode != "none":
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

        # Light logging of effective alpha and batch popularity when enabled
        if self.retain_pop_mode != "none":
            self.log({
                "ada_alpha_eff": float(alpha_eff),
                "ada_beta_mean_forget": float(pop_beta_mean) if pop_beta_mean is not None else 0.0,
            })
        return (loss, forget_outputs) if return_outputs else loss

    # ------------------------ Epoch-end Adaptation ------------------------
    @torch.no_grad()
    def _prepare_eval_subsets(self):
        """Collect small subsets from the underlying datasets for quick epoch-end evals."""
        self._retain_subset = None
        self._forget_subset = None

        # Train dataset is a ForgetRetainDataset
        train_ds = getattr(self, "train_dataset", None)
        if train_ds is None:
            return
        retain_ds = getattr(train_ds, "retain", None)
        forget_ds = getattr(train_ds, "forget", None)
        if retain_ds is not None and len(retain_ds) > 0:
            n = min(self.retain_eval_samples, len(retain_ds))
            self._retain_subset = [retain_ds[i] for i in range(n)]
        if forget_ds is not None and len(forget_ds) > 0:
            n = min(self.forget_eval_samples, len(forget_ds))
            self._forget_subset = [forget_ds[i] for i in range(n)]

    @torch.no_grad()
    def _batch_retain_loss(self, model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Filter out non-model kwargs like 'pop_sum', 'index'
        safe_batch = {
            k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")
        }
        if self.retain_loss_type == "NLL":
            outputs = model(**safe_batch)
            return outputs.loss
        elif self.retain_loss_type == "KL":
            kl, _ = compute_kl_divergence(model, self.ref_model, safe_batch)
            return kl
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _compute_retain_loss(self, model) -> float:
        if not self._retain_subset:
            return 0.0
        dl = DataLoader(self._retain_subset, batch_size=8, collate_fn=self.data_collator)
        total = 0.0
        count = 0
        device = self.accelerator.device
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = self._batch_retain_loss(model, batch)
            total += float(loss.detach().item())
            count += 1
        return total / max(1, count)

    @torch.no_grad()
    def _compute_forget_loss(self) -> Optional[float]:
        if not self._forget_subset:
            return None
        dl = DataLoader(self._forget_subset, batch_size=8, collate_fn=self.data_collator)
        total = 0.0
        count = 0
        device = self.accelerator.device
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = compute_wga_loss_dynamic_beta(self.model, batch, beta_from_pop_sum=True, rep_coeff=self.rep_coeff)
            total += float(loss.detach().item())
            count += 1
        return total / max(1, count)

    @torch.no_grad()
    def post_epoch_update(self):
        # Compute retain degradation
        L_ret_cur = self._compute_retain_loss(self.model)
        if self.retain_loss_type == "NLL":
            delta_rel = max(0.0, (L_ret_cur - self.L_ret_ref) / max(self.L_ret_ref, 1e-8))
        else:  # KL: treat directly as a positive deviation to be constrained
            delta_rel = max(0.0, L_ret_cur)

        # Dead-zone band in relative units
        upper = self.eps + self.tau

        # PI control on the error e = delta_rel - eps
        e = delta_rel - self.eps
        if abs(e) <= self.tau:
            # Inside deadband: slight integral decay to avoid drift
            self._pi_err_int *= 0.9
        else:
            # Update integral with anti-windup
            self._pi_err_int = max(-self.lambda_max, min(self.lambda_max, self._pi_err_int + e))
            # Proportional + integral update; project to feasible region
            update = self.kp * e + self.ki * self._pi_err_int
            self.lambda_k = max(0.0, min(self.lambda_max, self.lambda_k + update))

        # Early stopping heuristic
        curr_forget = self._compute_forget_loss()
        bad = False
        if curr_forget is not None and self.prev_forget_loss is not None:
            improve = self.prev_forget_loss - curr_forget  # forget loss should decrease (more negative)
            # Treat absolute change: if improvement small and retain degradation high
            if (improve < self.forget_improve_eps) and (delta_rel > upper):
                bad = True
        self.prev_forget_loss = curr_forget
        if bad:
            self.prev_bad_epochs += 1
        else:
            self.prev_bad_epochs = 0

        # Log adaptation state
        self.log(
            {
                "ada_lambda": float(self.lambda_k),
                "ada_alpha_k": float(self.alpha0 + self.lambda_k),
                "ada_gamma_k": float(self.gamma_k),
                "ada_delta_k": float(delta_rel),
                "ada_Lret_ref": float(self.L_ret_ref),
                "ada_Lret_cur": float(L_ret_cur),
            }
        )

        # Handle early stopping request
        if self.prev_bad_epochs >= self.consecutive_patience:
            # Signal HF Trainer to stop after this epoch
            if hasattr(self, "control"):
                self.control.should_training_stop = True

    # Hook into epoch end via callback-like method
    def _maybe_post_epoch(self):
        # Called by callback at epoch end
        self.post_epoch_update()

class AdaWGDCallback(TrainerCallback):
    def __init__(self, trainer: AdaWGD):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Expose control back to trainer for early stop
        self.trainer.control = control
        self.trainer._maybe_post_epoch()
