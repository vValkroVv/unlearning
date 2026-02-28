from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from trainer.unlearn.npo import NPO


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
        sam_rho: float = 0.05,
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
        grads: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        if not params:
            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        device = params[0].device
        sq_sum = torch.zeros((), device=device, dtype=torch.float32)
        for p, g in zip(params, grads):
            if g is None:
                continue
            grad = g
            if self.sam_adaptive:
                grad = p.detach().abs() * grad
            sq_sum = sq_sum + (grad.float() * grad.float()).sum()
        return torch.sqrt(sq_sum)

    @torch.no_grad()
    def _perturb_weights(
        self,
        params: List[torch.nn.Parameter],
        grads: List[Optional[torch.Tensor]],
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

            e_w = (perturb * scale.to(perturb.dtype)).to(p.dtype)
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
        second_pass_grads: List[Optional[torch.Tensor]],
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

        with self.compute_loss_context_manager():
            loss_1 = self.compute_loss(model, inputs, return_outputs=False)
        grads_1 = torch.autograd.grad(
            loss_1,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_norm = self._grad_norm(params, list(grads_1))
        sam_state = self._perturb_weights(params, list(grads_1), grad_norm)

        with self.compute_loss_context_manager():
            loss_2 = self.compute_loss(model, inputs, return_outputs=False)
        grads_2 = torch.autograd.grad(
            loss_2,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        self._restore_weights(params, sam_state)
        self._set_final_grads(params, list(grads_2), prev_grads, grad_scale)

        try:
            self.log(
                {
                    "npo_sam_loss1": float(loss_1.detach().item()),
                    "npo_sam_loss2": float(loss_2.detach().item()),
                    "npo_sam_grad_norm": float(grad_norm.detach().item()),
                    "npo_sam_rho": float(self.sam_rho),
                    "npo_sam_adaptive": 1.0 if self.sam_adaptive else 0.0,
                }
            )
        except Exception:
            pass

        return loss_2.detach() * grad_scale
