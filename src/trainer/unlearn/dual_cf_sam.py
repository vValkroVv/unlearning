from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from trainer.unlearn.dual_cf import DualCF


@dataclass
class _SAMState:
    e_ws: List[Optional[torch.Tensor]]


class DualCFSAM(DualCF):
    def __init__(
        self,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
        sam_eps: float = 1e-12,
        sam_risk_threshold: float = 0.75,
        sam_start_epoch: float = 2.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        self.sam_eps = float(sam_eps)
        self.sam_risk_threshold = float(sam_risk_threshold)
        self.sam_start_epoch = float(sam_start_epoch)

    def _trainable_params(self, model: torch.nn.Module) -> List[torch.nn.Parameter]:
        return [param for param in model.parameters() if param.requires_grad]

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
            scaled = param.detach().abs() * grad if self.sam_adaptive else grad
            sq_sum = sq_sum + (scaled.float() * scaled.float()).sum()
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
    def _restore_weights(
        self, params: List[torch.nn.Parameter], state: _SAMState
    ) -> None:
        for param, e_w in zip(params, state.e_ws):
            if e_w is None:
                continue
            param.sub_(e_w)

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

    def _current_batch_risk(self, forget_inputs) -> float:
        batch_size = int(forget_inputs["original"]["input_ids"].shape[0])
        _, _, _, risk_gate = self._routing_state(
            forget_inputs=forget_inputs,
            device=self.accelerator.device,
            batch_size=batch_size,
        )
        return float(self._summarize_risk(risk_gate).detach().item())

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[DualCFSAM] DeepSpeed is not supported in this integration."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[DualCFSAM] Multi-process training is not supported in this integration."
            )

        epoch = float(getattr(self.state, "epoch", 0.0) or 0.0)
        batch_risk = self._current_batch_risk(inputs["forget"])
        use_sam = epoch >= self.sam_start_epoch and batch_risk >= self.sam_risk_threshold
        if not use_sam:
            return super().training_step(model, inputs)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        params = self._trainable_params(model)
        if not params:
            raise RuntimeError("[DualCFSAM] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

        with self.compute_loss_context_manager():
            forget_vec_1, _, aux_1 = self._compute_forget_loss_vec(model, inputs["forget"])
            forget_loss_1 = forget_vec_1.mean()
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
                forget_vec_2, _, aux_2 = self._compute_forget_loss_vec(
                    model,
                    inputs["forget"],
                )
                forget_loss_2 = forget_vec_2.mean()
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
            retain_loss = self.compute_retain_loss(
                model=model,
                retain_inputs=self._retain_inputs(inputs),
            )
        retain_grads = torch.autograd.grad(
            retain_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        alpha_eff, lambda_ret_batch, risk_batch = self._compute_alpha_eff(
            aux_2["risk_gate"]
        )
        combined_grads: List[Optional[torch.Tensor]] = []
        for forget_grad, retain_grad in zip(forget_grads, retain_grads):
            grad = None
            if forget_grad is not None:
                grad = self.gamma * forget_grad
            if retain_grad is not None:
                retain_component = alpha_eff * retain_grad
                grad = retain_component if grad is None else grad + retain_component
            combined_grads.append(grad)

        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)

        try:
            self.log(
                {
                    "dualcf_sam_used": 1.0,
                    "dualcf_sam_batch_risk": float(batch_risk),
                    "dualcf_sam_forget_loss_1": float(forget_loss_1.detach().item()),
                    "dualcf_sam_forget_loss_2": float(forget_loss_2.detach().item()),
                    "dualcf_sam_retain_loss": float(retain_loss.detach().item()),
                    "dualcf_sam_grad_norm": float(grad_norm.detach().item()),
                    "dualcf_sam_rho": float(self.sam_rho),
                    "dualcf_sam_adaptive": 1.0 if self.sam_adaptive else 0.0,
                    "dualcf_sam_alpha_eff": float(alpha_eff.detach().item()),
                    "dualcf_lambda_ret_batch": float(lambda_ret_batch.detach().item()),
                    "dualcf_risk_batch": float(risk_batch.detach().item()),
                    "dualcf_lambda_belief_mean": float(
                        aux_2["lambda_belief"].mean().detach().item()
                    ),
                    "dualcf_neg_mask_frac": float(aux_2["neg_mask_frac"].detach().item()),
                }
            )
        except Exception:
            pass

        total_loss = self.gamma * forget_loss_2 + alpha_eff * retain_loss
        return total_loss.detach() * grad_scale
