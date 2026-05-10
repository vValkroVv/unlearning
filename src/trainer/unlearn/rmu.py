"""RMU activation-steering unlearning trainer.

Adapted from the WMDP / RMU implementation, with repo-native retain-loss,
PEFT-safe module matching, and LoRA-safe parameter selection.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

import torch
from trainer.unlearn.grad_diff import GradDiff

try:  # deepspeed is installed in the production env, but keep import robust.
    import deepspeed
except Exception:  # pragma: no cover - import guard for CPU/lightweight smoke envs.
    deepspeed = None


logger = logging.getLogger(__name__)


class RMU(GradDiff):
    def __init__(
        self,
        module_regex: str = r"model\.layers\.7",
        trainable_params_regex: Sequence[str] | str = (
            r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight",
        ),
        steering_coeff: float = 20.0,
        *args,
        **kwargs,
    ):
        """RMU trainer.

        Args:
            module_regex: Regex that should match exactly one hidden-state module.
                For PEFT/LoRA models use a suffix-safe regex such as
                ``.*layers\\.7$``.
            trainable_params_regex: Regex or list of regexes selecting parameters
                for the optimizer. For LoRA comparison runs use
                ``.*lora_[AB].*``.
            steering_coeff: Norm multiplier for the random RMU control vector.
        """
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.module_regex = str(module_regex)
        self.trainable_params_regex = self._normalize_regexes(trainable_params_regex)
        self.steering_coeff = float(steering_coeff)
        self.control_vec: torch.Tensor | None = None

        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)

    @staticmethod
    def _normalize_regexes(regexes: Sequence[str] | str) -> list[str]:
        if isinstance(regexes, str):
            return [regexes]
        return [str(pattern) for pattern in regexes]

    def create_optimizer(self):
        # Freeze everything, then enable only the RMU-selected parameters before
        # Hugging Face Trainer builds the optimizer. Do not re-enable all params
        # afterwards: that computes useless base-model gradients in LoRA runs.
        self._freeze_all_params(self.model, False)
        matched = self._set_trainable_params(
            self.model,
            self.trainable_params_regex,
            True,
        )
        if matched == 0:
            raise ValueError(
                "[RMU] No trainable parameters matched "
                f"trainable_params_regex={self.trainable_params_regex}. "
                "For LoRA runs use e.g. ['.*lora_[AB].*']; for full-weight RMU "
                "use a base-model parameter regex."
            )
        logger.info("[RMU] Enabled %d trainable parameter tensors.", matched)
        return super().create_optimizer()

    def _unwrap_model(self, model):
        if deepspeed is not None and isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
        if hasattr(model, "module") and model.module is not model:
            return model.module
        return model

    def _get_matching_module(self, model, module_regex: str):
        """Return the single module matching ``module_regex``.

        Matching uses ``re.fullmatch`` deliberately. This prevents accidentally
        matching both a decoder layer and all of its child modules. Use a regex
        such as ``.*layers\\.7$`` for PEFT prefixes like
        ``base_model.model.model.layers.7``.
        """
        model = self._unwrap_model(model)
        pattern = re.compile(module_regex)
        matched_modules = {
            name: module
            for name, module in model.named_modules()
            if pattern.fullmatch(name)
        }

        if len(matched_modules) > 1:
            raise ValueError(
                f"[RMU] More than one module matched {module_regex}: "
                f"{list(matched_modules.keys())[:20]}"
            )
        if not matched_modules:
            layer_like = [name for name, _ in model.named_modules() if "layers" in name]
            hint = ", ".join(layer_like[:20])
            raise ValueError(
                f"[RMU] No module matched module_regex={module_regex}. "
                "For LoRA/PEFT models try module_regex='.*layers\\.7$'. "
                f"Layer-like module examples: {hint}"
            )

        name, module = next(iter(matched_modules.items()))
        logger.info("[RMU] Matched module %s for regex %s", name, module_regex)
        return module

    def _freeze_all_params(self, model, requires_grad: bool = True) -> None:
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(
        self,
        model,
        trainable_params_regex: Sequence[str],
        requires_grad: bool = True,
    ) -> int:
        patterns = [re.compile(pattern) for pattern in trainable_params_regex]
        matched = 0
        for name, param in model.named_parameters():
            if any(pattern.fullmatch(name) for pattern in patterns):
                param.requires_grad = requires_grad
                matched += 1
        return matched

    def forward_with_cache(self, model, inputs, module, no_grad: bool = True):
        cache: list[torch.Tensor] = []

        def hook(_module, _input, output):
            cache.append(output[0] if isinstance(output, tuple) else output)
            return None

        hook_handle = module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(not no_grad):
                outputs = model(**inputs)
        finally:
            hook_handle.remove()

        if not cache:
            raise RuntimeError("[RMU] Forward hook did not capture activations.")
        return cache[0], outputs

    def get_control_vector(
        self,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.control_vec is None or self.control_vec.shape[-1] != dim:
            random_vector = torch.rand(1, 1, dim, device=device, dtype=torch.float32)
            self.control_vec = random_vector / torch.norm(random_vector).clamp_min(1e-12)
            self.control_vec = self.control_vec * self.steering_coeff
        return self.control_vec.to(device=device, dtype=dtype)

    def compute_activation_loss(
        self,
        activation1: torch.Tensor,
        activation2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squared_diff = torch.nn.functional.mse_loss(
            activation1,
            activation2,
            reduction="none",
        )
        mask = mask.to(device=squared_diff.device, dtype=squared_diff.dtype)
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        per_token_diff = (squared_diff * expanded_mask).mean(dim=-1)
        token_counts = mask.sum(dim=-1).clamp_min(1.0)
        return (per_token_diff.sum(dim=-1) / token_counts).mean()

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type != "EMBED_DIFF":
            return super().compute_retain_loss(model, retain_inputs)

        model_retain_activations, _ = self.forward_with_cache(
            model,
            retain_inputs,
            module=self.model_module,
            no_grad=False,
        )
        ref_retain_activations, _ = self.forward_with_cache(
            self.ref_model,
            retain_inputs,
            module=self.ref_module,
            no_grad=True,
        )
        mask = retain_inputs["labels"] != -100
        return self.compute_activation_loss(
            model_retain_activations,
            ref_retain_activations.to(model_retain_activations.device),
            mask,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model,
            forget_inputs,
            self.model_module,
            no_grad=False,
        )
        control_vec = self.get_control_vector(
            model_forget_activations.shape[-1],
            device=model_forget_activations.device,
            dtype=model_forget_activations.dtype,
        ).expand_as(model_forget_activations)
        forget_mask = forget_inputs["labels"] != -100
        forget_loss = self.compute_activation_loss(
            model_forget_activations,
            control_vec,
            forget_mask,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        try:
            self.log(
                {
                    "rmu_forget_loss": float(forget_loss.detach().item()),
                    "rmu_retain_loss": float(retain_loss.detach().item()),
                    "rmu_total_loss": float(loss.detach().item()),
                    "rmu_activation_norm": float(
                        model_forget_activations.detach().norm(dim=-1).mean().item()
                    ),
                    "rmu_control_norm": float(control_vec.detach().norm(dim=-1).mean().item()),
                    "rmu_steering_coeff": float(self.steering_coeff),
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss
