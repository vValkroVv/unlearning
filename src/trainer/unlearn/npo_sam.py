from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss


@dataclass
class _SAMState:
    e_ws: List[Optional[torch.Tensor]]


class NPOSAM(NPO):
    """
    NPO + SAM (Sharpness-Aware Minimization), as used in Unlearn-Smooth.

    SAM is implemented with a two-pass update:
    1) compute gradients at current weights, then perturb parameters by +e(w)
    2) compute gradients at perturbed weights (actual update gradients)
    3) restore original weights
    """

    def __init__(
        self,
        sam_rho: float = 0.01,
        sam_adaptive: bool = False,
        sam_eps: float = 1e-12,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        self.sam_eps = float(sam_eps)

    def _trainable_params(self, model: torch.nn.Module) -> List[torch.nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    def _stash_grads(self, params: List[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
        stashed: List[Optional[torch.Tensor]] = []
        for p in params:
            if p.grad is None:
                stashed.append(None)
            else:
                stashed.append(p.grad.detach().clone())
        return stashed

    def _clear_grads_set_to_none(self, params: List[torch.nn.Parameter]) -> None:
        for p in params:
            p.grad = None

    @torch.no_grad()
    def _grad_norm(
        self,
        params: List[torch.nn.Parameter],
        grads: Sequence[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        if not params:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        ref_device = None
        for g in grads:
            if g is not None:
                ref_device = g.device
                break
        if ref_device is None:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        sq_sum = torch.zeros((), device=ref_device, dtype=torch.float32)
        for p, g in zip(params, grads):
            if g is None:
                continue
            grad = g
            if self.sam_adaptive:
                grad = p.detach().abs() * grad
            grad_sq = grad.float()
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

        for p, g in zip(params, grads):
            if g is None:
                e_ws.append(None)
                continue

            if self.sam_adaptive:
                perturb = p.detach().abs() * g
            else:
                perturb = g

            scale_t = scale.to(device=perturb.device, dtype=perturb.dtype)
            e_w = (perturb * scale_t).to(dtype=p.dtype)
            p.add_(e_w)
            e_ws.append(e_w)

        return _SAMState(e_ws=e_ws)

    @torch.no_grad()
    def _restore_weights(
        self, params: List[torch.nn.Parameter], state: _SAMState
    ) -> None:
        for p, e_w in zip(params, state.e_ws):
            if e_w is None:
                continue
            p.sub_(e_w)

    def _set_final_grads(
        self,
        params: List[torch.nn.Parameter],
        second_pass_grads: Sequence[Optional[torch.Tensor]],
        prev_grads: List[Optional[torch.Tensor]],
        grad_scale: float,
    ) -> None:
        for p, g2, g_prev in zip(params, second_pass_grads, prev_grads):
            grad = None
            if g2 is not None:
                grad = g2.detach() * grad_scale

            if g_prev is not None:
                if grad is None:
                    grad = g_prev
                else:
                    grad = grad + g_prev

            p.grad = grad

    def _compute_forget_loss_only(self, model, inputs):
        forget_inputs = inputs["forget"]
        if isinstance(forget_inputs, dict) and "original" in forget_inputs:
            forget_inputs = forget_inputs["original"]

        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        return forget_loss

    def _compute_retain_loss_only(self, model, inputs):
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[NPOSAM] DeepSpeed is not supported in this integration."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[NPOSAM] Multi-process training is not supported in this integration."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        params = self._trainable_params(model)
        if not params:
            raise RuntimeError("[NPOSAM] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

        # 1) Forget pass at current weights (for SAM perturbation direction).
        with self.compute_loss_context_manager():
            forget_loss_1 = self._compute_forget_loss_only(model, inputs)
        grads_1 = torch.autograd.grad(
            forget_loss_1,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_norm = self._grad_norm(params, grads_1)
        sam_state = self._perturb_weights(params, grads_1, grad_norm)

        # 2) Forget pass at perturbed weights (SAM second pass).
        try:
            self._clear_grads_set_to_none(params)
            with self.compute_loss_context_manager():
                forget_loss_2 = self._compute_forget_loss_only(model, inputs)
            forget_grads = torch.autograd.grad(
                forget_loss_2,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        finally:
            # Restore weights even if an exception occurs.
            self._restore_weights(params, sam_state)

        # 3) Retain gradient at restored (unperturbed) weights.
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

        # 4) Combine forget and retain gradients with NPO weights.
        combined_grads: List[Optional[torch.Tensor]] = []
        for g_forget, g_retain in zip(forget_grads, retain_grads):
            grad = None
            if g_forget is not None:
                grad = self.gamma * g_forget
            if g_retain is not None:
                retain_component = self.alpha * g_retain
                grad = retain_component if grad is None else grad + retain_component
            combined_grads.append(grad)

        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)

        try:
            self.log(
                {
                    "npo_sam_forget_loss_1": float(forget_loss_1.detach().item()),
                    "npo_sam_forget_loss_2": float(forget_loss_2.detach().item()),
                    "npo_sam_retain_loss": float(retain_loss.detach().item()),
                    "npo_sam_grad_norm": float(grad_norm.detach().item()),
                    "npo_sam_rho": float(self.sam_rho),
                    "npo_sam_adaptive": 1.0 if self.sam_adaptive else 0.0,
                }
            )
        except Exception:
            pass

        total_loss = self.gamma * forget_loss_2 + self.alpha * retain_loss
        return total_loss.detach() * grad_scale
