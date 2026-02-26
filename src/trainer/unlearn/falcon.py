import re
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff

try:
    import deepspeed  # type: ignore

    _HAS_DEEPSPEED = True
except Exception:
    deepspeed = None
    _HAS_DEEPSPEED = False


class FALCON(GradDiff):
    """
    FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment.

    Integration choices for this repository:
    - Uses token-level samples (labels != -100) to make contrastive losses meaningful
      with per_device_train_batch_size=1.
    - Exposes target_layer directly (manual substitute for MI-guided layer selection).
    - Applies orthogonal gradient projection when cosine(g_forget, g_retain) is below
      conflict_cos_threshold.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        temperature: float = 0.07,
        k_svd: int = 16,
        pov_alpha: float = 1.0,
        pov_noise_std: float = 0.0,
        pov_transform: str = "tanh",
        target_layer: int = 7,
        conflict_cos_threshold: float = 0.0,
        retain_mode: str = "contrastive",
        *args,
        **kwargs,
    ):
        # Keep GradDiff plumbing for gamma/alpha and ref model preparation helpers.
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="NLL", *args, **kwargs)

        self.temperature = float(temperature)
        self.k_svd = int(k_svd)
        self.pov_alpha = float(pov_alpha)
        self.pov_noise_std = float(pov_noise_std)
        self.pov_transform = str(pov_transform).lower()
        self.target_layer = int(target_layer)
        self.conflict_cos_threshold = float(conflict_cos_threshold)
        self.retain_mode = str(retain_mode).lower()

        model_unwrapped = self._unwrap(self.model)
        self.uses_lora = hasattr(model_unwrapped, "disable_adapter")

        self.ref_model = None
        if not self.uses_lora:
            self.ref_model = self._prepare_ref_model(self.model)

        self.model_module = self._find_layer_module(self.model, self.target_layer)
        ref_for_hook = self.ref_model if self.ref_model is not None else self.model
        self.ref_module = self._find_layer_module(ref_for_hook, self.target_layer)

    def _unwrap(self, model):
        if _HAS_DEEPSPEED and isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
        if hasattr(model, "module"):
            return model.module
        return model

    def _find_layer_module(self, model, layer_idx: int):
        model_unwrapped = self._unwrap(model)
        candidates = []
        for name, module in model_unwrapped.named_modules():
            parts = name.split(".")
            if len(parts) < 2:
                continue
            if parts[-1] != str(layer_idx):
                continue
            if parts[-2] in {"layers", "h", "blocks"}:
                candidates.append((name, module))

        if not candidates:
            fallback_patterns = (
                rf".*model\.layers\.{layer_idx}$",
                rf".*model\.model\.layers\.{layer_idx}$",
                rf".*base_model\.model\.layers\.{layer_idx}$",
                rf".*base_model\.model\.model\.layers\.{layer_idx}$",
            )
            for name, module in model_unwrapped.named_modules():
                if any(re.match(pat, name) for pat in fallback_patterns):
                    candidates.append((name, module))

        if not candidates:
            raise ValueError(
                f"[FALCON] Could not find module for target_layer={layer_idx}. "
                "Inspect model.named_modules() for your architecture."
            )

        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][1]

    def _as_model_inputs(self, inputs):
        allowed = {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "token_type_ids",
            "inputs_embeds",
        }
        return {k: v for k, v in inputs.items() if k in allowed}

    def _forward_with_cache(self, model, inputs, module, no_grad: bool):
        cache = []

        def hook(_module, _inp, out):
            cache.append(out[0] if isinstance(out, tuple) else out)
            return None

        handle = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        handle.remove()
        if not cache:
            raise RuntimeError("[FALCON] Forward hook did not capture activations.")
        return cache[0], outputs

    @contextmanager
    def _frozen_forward_context(self):
        if self.ref_model is not None:
            yield self.ref_model
            return

        model_unwrapped = self._unwrap(self.model)
        disable_adapter = getattr(model_unwrapped, "disable_adapter", None)
        if disable_adapter is None:
            yield self.model
            return

        with disable_adapter():
            yield self.model

    def _token_samples(
        self,
        acts: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if labels is not None:
            mask = labels != -100
            if mask.any():
                return acts[mask]
        if attention_mask is not None:
            mask = attention_mask.bool()
            if mask.any():
                return acts[mask]
        return acts[:, -1, :]

    def _compute_povs(self, ref_samples: torch.Tensor, num_samples: int) -> torch.Tensor:
        X = ref_samples.detach()
        X = X - X.mean(dim=0, keepdim=True)
        n, d = X.shape
        k = max(1, min(self.k_svd, n, d))

        Xf = X.float()
        try:
            _u, _s, vh = torch.linalg.svd(Xf, full_matrices=False)
            V = vh[:k].transpose(0, 1).contiguous()
        except RuntimeError:
            V = torch.pca_lowrank(Xf, q=k, center=False)[2]

        V = V.to(device=ref_samples.device, dtype=ref_samples.dtype)
        r = torch.randn(
            (num_samples, d), device=ref_samples.device, dtype=ref_samples.dtype
        )

        proj = (r @ V) @ V.transpose(0, 1)
        pov = r - self.pov_alpha * proj

        if self.pov_transform == "tanh":
            pov = torch.tanh(pov)
        elif self.pov_transform == "none":
            pass
        else:
            raise ValueError(f"[FALCON] Unsupported pov_transform={self.pov_transform}")

        if self.pov_noise_std > 0:
            pov = pov + self.pov_noise_std * torch.randn_like(pov)

        return F.normalize(pov, dim=-1)

    def _forget_infonce(
        self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positives, dim=-1)
        n = F.normalize(negatives, dim=-1)

        pos_sim = (a * p).sum(dim=-1, keepdim=True)
        neg_sim = a @ n.transpose(0, 1)
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
        return F.cross_entropy(logits.float(), labels)

    def _retain_alignment_loss(self, upd: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.retain_mode == "mse":
            return F.mse_loss(upd, ref)
        if self.retain_mode != "contrastive":
            raise ValueError(f"[FALCON] Unsupported retain_mode={self.retain_mode}")

        a = F.normalize(upd, dim=-1)
        p = F.normalize(ref, dim=-1)
        n = a.shape[0]
        if n <= 1:
            return (1.0 - (a * p).sum(dim=-1)).mean()
        logits = (a @ p.transpose(0, 1)) / self.temperature
        labels = torch.arange(n, device=a.device)
        return F.cross_entropy(logits.float(), labels)

    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "[FALCON] DeepSpeed path is not implemented for manual gradient projection."
            )
        if getattr(self.accelerator, "num_processes", 1) > 1:
            raise NotImplementedError(
                "[FALCON] Multi-process training is not implemented for manual gradient projection."
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        forget_inputs = self._as_model_inputs(inputs["forget"])
        retain_inputs = self._as_model_inputs(inputs["retain"])

        with self.compute_loss_context_manager():
            upd_f_acts, _ = self._forward_with_cache(
                model, forget_inputs, module=self.model_module, no_grad=False
            )
            with self._frozen_forward_context() as frozen_model:
                ref_f_acts, _ = self._forward_with_cache(
                    frozen_model, forget_inputs, module=self.ref_module, no_grad=True
                )

            upd_f = self._token_samples(
                upd_f_acts, forget_inputs.get("labels"), forget_inputs.get("attention_mask")
            )
            ref_f = self._token_samples(
                ref_f_acts, forget_inputs.get("labels"), forget_inputs.get("attention_mask")
            )
            povs = self._compute_povs(ref_f, num_samples=upd_f.shape[0])
            forget_loss = self._forget_infonce(anchor=upd_f, positives=povs, negatives=ref_f)

            upd_r_acts, _ = self._forward_with_cache(
                model, retain_inputs, module=self.model_module, no_grad=False
            )
            with self._frozen_forward_context() as frozen_model:
                ref_r_acts, _ = self._forward_with_cache(
                    frozen_model, retain_inputs, module=self.ref_module, no_grad=True
                )

            upd_r = self._token_samples(
                upd_r_acts, retain_inputs.get("labels"), retain_inputs.get("attention_mask")
            )
            ref_r = self._token_samples(
                ref_r_acts, retain_inputs.get("labels"), retain_inputs.get("attention_mask")
            )
            retain_loss = self._retain_alignment_loss(upd=upd_r, ref=ref_r.to(upd_r.device))

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("[FALCON] No trainable parameters found.")

        g_forget = torch.autograd.grad(
            forget_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        g_retain = torch.autograd.grad(
            retain_loss,
            trainable_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        dot = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
        nf = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
        nr = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
        for gf, gr in zip(g_forget, g_retain):
            if gf is None or gr is None:
                continue
            gf32 = gf.float()
            gr32 = gr.float()
            dot = dot + (gf32 * gr32).sum()
            nf = nf + (gf32 * gf32).sum()
            nr = nr + (gr32 * gr32).sum()

        eps = 1e-12
        cos = dot / (torch.sqrt(nf + eps) * torch.sqrt(nr + eps) + eps)
        conflict = bool(
            cos.detach().item() < self.conflict_cos_threshold and nr.detach().item() > 0.0
        )
        proj_coeff = (dot / (nr + eps)) if conflict else None

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        grad_scale = 1.0 / grad_acc_steps

        for param, gf, gr in zip(trainable_params, g_forget, g_retain):
            if gf is None and gr is None:
                continue
            if gf is None:
                gf = torch.zeros_like(gr)
            if gr is None:
                gr = torch.zeros_like(gf)

            if conflict and proj_coeff is not None:
                gf = gf - proj_coeff.to(gf.dtype) * gr

            grad = (self.gamma * gf + self.alpha * gr).detach() * grad_scale
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad)

        total_loss = self.gamma * forget_loss + self.alpha * retain_loss

        try:
            self.log(
                {
                    "falcon_forget_loss": float(forget_loss.detach().item()),
                    "falcon_retain_loss": float(retain_loss.detach().item()),
                    "falcon_grad_cos": float(cos.detach().item()),
                    "falcon_conflict": 1.0 if conflict else 0.0,
                }
            )
        except Exception:
            pass

        return total_loss.detach() * grad_scale
