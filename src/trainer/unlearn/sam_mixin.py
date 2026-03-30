from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class _SAMState:
    e_ws: List[Optional[torch.Tensor]]


class ManualGradMixin:
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


class SAMMixin(ManualGradMixin):
    SAM_LOG_PREFIX = "sam"

    def _manual_retain_weight(self, inputs, components=None):
        del inputs, components
        return self.alpha

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

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                f"[{self.__class__.__name__}] DeepSpeed is not supported in this integration."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                f"[{self.__class__.__name__}] Multi-process training is not supported in this integration."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        params = self._trainable_params(model)
        if not params:
            raise RuntimeError(f"[{self.__class__.__name__}] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps
        self._last_manual_components = None

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

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

        retain_weight = self._manual_retain_weight(
            inputs=inputs,
            components=getattr(self, "_last_manual_components", None),
        )
        retain_weight_value = (
            float(retain_weight.detach().item())
            if torch.is_tensor(retain_weight)
            else float(retain_weight)
        )

        combined_grads: List[Optional[torch.Tensor]] = []
        for g_forget, g_retain in zip(forget_grads, retain_grads):
            grad = None
            if g_forget is not None:
                grad = self.gamma * g_forget
            if g_retain is not None:
                if torch.is_tensor(retain_weight):
                    retain_scale = retain_weight.to(
                        device=g_retain.device,
                        dtype=g_retain.dtype,
                    )
                else:
                    retain_scale = retain_weight
                retain_component = retain_scale * g_retain
                grad = retain_component if grad is None else grad + retain_component
            combined_grads.append(grad)

        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)

        if torch.is_tensor(retain_weight):
            retain_weight_loss = retain_weight.to(
                device=retain_loss.device,
                dtype=retain_loss.dtype,
            )
        else:
            retain_weight_loss = retain_weight
        total_loss = self.gamma * forget_loss_2 + retain_weight_loss * retain_loss
        prefix = str(getattr(self, "SAM_LOG_PREFIX", self.__class__.__name__.lower()))
        extra_logs = {
            f"{prefix}_forget_loss_1": float(forget_loss_1.detach().item()),
            f"{prefix}_forget_loss_2": float(forget_loss_2.detach().item()),
            f"{prefix}_retain_loss": float(retain_loss.detach().item()),
            f"{prefix}_retain_weight": retain_weight_value,
            f"{prefix}_grad_norm": float(grad_norm.detach().item()),
            f"{prefix}_rho": float(self.sam_rho),
            f"{prefix}_adaptive": 1.0 if self.sam_adaptive else 0.0,
        }
        try:
            components = getattr(self, "_last_manual_components", None)
            if components is not None and hasattr(self, "_maybe_log"):
                self._maybe_log(
                    components=components,
                    retain_loss=retain_loss,
                    loss=total_loss,
                    extra_logs=extra_logs,
                )
            else:
                self.log(extra_logs)
        except Exception:
            pass

        return total_loss.detach() * grad_scale
