import torch

from trainer.unlearn.sam_mixin import ManualGradMixin
from trainer.unlearn.span_cf_simnpo import SpanCFSimNPO


class SpanCFSimNPOProjected(ManualGradMixin, SpanCFSimNPO):
    LOG_PREFIX = "span_simnpo_proj"

    def __init__(
        self,
        projection_cos_threshold=0.0,
        projection_eps=1e-12,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.projection_cos_threshold = float(projection_cos_threshold)
        self.projection_eps = float(projection_eps)

    def _resolve_projection(self, neg_grads, retain_grads, device):
        dot = torch.zeros((), device=device, dtype=torch.float32)
        neg_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        retain_norm_sq = torch.zeros((), device=device, dtype=torch.float32)

        for g_neg, g_retain in zip(neg_grads, retain_grads):
            if g_neg is None or g_retain is None:
                continue
            neg_fp32 = g_neg.float()
            retain_fp32 = g_retain.float()
            dot = dot + (neg_fp32 * retain_fp32).sum()
            neg_norm_sq = neg_norm_sq + (neg_fp32 * neg_fp32).sum()
            retain_norm_sq = retain_norm_sq + (retain_fp32 * retain_fp32).sum()

        eps = float(self.projection_eps)
        cos = dot / (
            torch.sqrt(neg_norm_sq + eps) * torch.sqrt(retain_norm_sq + eps) + eps
        )
        cos_value = float(cos.detach().item())
        has_retain_grad = float(retain_norm_sq.detach().item()) > 0.0
        conflict = bool(cos_value < self.projection_cos_threshold and has_retain_grad)
        coeff = (dot / (retain_norm_sq + eps)) if conflict else None
        return conflict, cos_value, coeff

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[SpanCFSimNPOProjected] DeepSpeed is not supported in this integration."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[SpanCFSimNPOProjected] Multi-process training is not supported in this integration."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        params = self._trainable_params(model)
        if not params:
            raise RuntimeError("[SpanCFSimNPOProjected] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

        with self.compute_loss_context_manager():
            components = self._compute_core_components(model, inputs)
            cf_loss = components["cf_loss"]
            neg_loss = components["neg_loss"]

        cf_grads = torch.autograd.grad(
            cf_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        neg_grads = torch.autograd.grad(
            neg_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

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

        conflict, cos_value, proj_coeff = self._resolve_projection(
            neg_grads=neg_grads,
            retain_grads=retain_grads,
            device=components["forget_loss"].device,
        )

        retain_weight = components["alpha_eff"]
        combined_grads = []
        for g_cf, g_neg, g_retain in zip(cf_grads, neg_grads, retain_grads):
            grad = None
            if g_cf is not None:
                grad = self.gamma * g_cf
            if g_neg is not None:
                if conflict and proj_coeff is not None and g_retain is not None:
                    g_neg = g_neg - proj_coeff.to(
                        device=g_neg.device,
                        dtype=g_neg.dtype,
                    ) * g_retain
                neg_component = self.gamma * g_neg
                grad = neg_component if grad is None else grad + neg_component
            if g_retain is not None:
                retain_component = retain_weight.to(
                    device=g_retain.device,
                    dtype=g_retain.dtype,
                ) * g_retain
                grad = retain_component if grad is None else grad + retain_component
            combined_grads.append(grad)

        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)

        total_loss = self.gamma * components["forget_loss"] + retain_weight * retain_loss
        conflict_ratio = 1.0 if conflict else 0.0
        try:
            self._maybe_log(
                components=components,
                retain_loss=retain_loss,
                loss=total_loss,
                extra_logs={
                    f"{self.log_prefix}_grad_cos": cos_value,
                    f"{self.log_prefix}_conflict_ratio": conflict_ratio,
                    f"{self.log_prefix}_conflict": 1.0 if conflict else 0.0,
                    f"{self.log_prefix}_projection_threshold": self.projection_cos_threshold,
                },
            )
        except Exception:
            pass

        return total_loss.detach() * grad_scale
