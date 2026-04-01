import torch

from trainer.unlearn.sam_mixin import SAMMixin
from trainer.unlearn.span_cf import SpanCF


class SpanCFSAMNPO(SAMMixin, SpanCF):
    LOG_PREFIX = "span_samnpo"
    SAM_LOG_PREFIX = "span_samnpo"

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

    def _compute_core_components(self, model, inputs):
        components = super()._compute_core_components(model, inputs)
        components["extra_logs"] = dict(components.get("extra_logs", {}))
        components["extra_logs"].update(
            {
                f"{self.log_prefix}_neg_branch_sam_only": 1.0,
                f"{self.log_prefix}_cf_branch_sam_only": 0.0,
            }
        )
        return components

    def _build_log_payload(self, components: dict, retain_loss, loss=None, extra_logs=None):
        payload = super()._build_log_payload(
            components=components,
            retain_loss=retain_loss,
            loss=loss,
            extra_logs=extra_logs,
        )
        prefix = self.log_prefix
        logged_neg_loss = components.get("logged_neg_loss", None)
        logged_forget_loss = components.get("logged_forget_loss", None)
        if logged_neg_loss is not None:
            payload[f"{prefix}_neg_loss"] = float(logged_neg_loss.detach().item())
        if logged_forget_loss is not None:
            payload[f"{prefix}_forget_loss"] = float(logged_forget_loss.detach().item())
        return payload

    def _compute_neg_loss_only(self, model, inputs):
        components = self._compute_core_components(model, inputs)
        return components["neg_loss"]

    def _compute_retain_loss_only(self, model, inputs):
        return self.compute_retain_loss(
            model=model,
            retain_inputs=self._retain_inputs(inputs),
        )

    def _manual_retain_weight(self, inputs, components=None):
        del inputs
        if components is None:
            return self.alpha
        return components["alpha_eff"]

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[SpanCFSAMNPO] DeepSpeed is not supported in this integration."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[SpanCFSAMNPO] Multi-process training is not supported in this integration."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        params = self._trainable_params(model)
        if not params:
            raise RuntimeError("[SpanCFSAMNPO] No trainable parameters found.")

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps
        self._last_manual_components = None

        prev_grads = self._stash_grads(params)
        self._clear_grads_set_to_none(params)

        with self.compute_loss_context_manager():
            components = self._compute_core_components(model, inputs)
            cf_loss = components["cf_loss"]
            neg_loss_1 = components["neg_loss"]

        cf_grads = torch.autograd.grad(
            cf_loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        neg_grads_1 = torch.autograd.grad(
            neg_loss_1,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_norm = self._grad_norm(params, neg_grads_1)
        sam_state = self._perturb_weights(params, neg_grads_1, grad_norm)

        try:
            self._clear_grads_set_to_none(params)
            with self.compute_loss_context_manager():
                neg_loss_2 = self._compute_neg_loss_only(model, inputs)
            neg_grads_2 = torch.autograd.grad(
                neg_loss_2,
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

        retain_weight = self._manual_retain_weight(inputs=inputs, components=components)
        retain_weight_value = (
            float(retain_weight.detach().item())
            if torch.is_tensor(retain_weight)
            else float(retain_weight)
        )

        combined_grads = []
        for g_cf, g_neg, g_retain in zip(cf_grads, neg_grads_2, retain_grads):
            grad = None
            if g_cf is not None:
                grad = self.gamma * g_cf
            if g_neg is not None:
                neg_component = self.gamma * g_neg
                grad = neg_component if grad is None else grad + neg_component
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
        forget_loss_update = cf_loss + neg_loss_2
        total_loss = self.gamma * forget_loss_update + retain_weight_loss * retain_loss

        components["logged_neg_loss"] = neg_loss_2
        components["logged_forget_loss"] = forget_loss_update
        self._last_manual_components = components
        extra_logs = {
            f"{self.log_prefix}_cf_loss_base": float(cf_loss.detach().item()),
            f"{self.log_prefix}_neg_loss_1": float(neg_loss_1.detach().item()),
            f"{self.log_prefix}_neg_loss_2": float(neg_loss_2.detach().item()),
            f"{self.log_prefix}_retain_loss": float(retain_loss.detach().item()),
            f"{self.log_prefix}_retain_weight": retain_weight_value,
            f"{self.log_prefix}_grad_norm": float(grad_norm.detach().item()),
            f"{self.log_prefix}_rho": float(self.sam_rho),
            f"{self.log_prefix}_adaptive": 1.0 if self.sam_adaptive else 0.0,
        }
        try:
            self._maybe_log(
                components=components,
                retain_loss=retain_loss,
                loss=total_loss,
                extra_logs=extra_logs,
            )
        except Exception:
            pass

        return total_loss.detach() * grad_scale
