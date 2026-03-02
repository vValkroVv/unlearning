import logging
import os
from typing import Dict, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


def canonicalize_weight_name(name: str) -> str:
    """Normalize parameter/module names to a stable base-weight identifier."""
    normalized = str(name)
    while normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    while normalized.startswith("base_model.model."):
        normalized = normalized[len("base_model.model.") :]
    normalized = normalized.replace(".base_layer.weight", ".weight")
    return normalized


def _weight_matches_targets(weight_name: str, target_modules: Sequence[str]) -> bool:
    return any(weight_name.endswith(f".{target}.weight") for target in target_modules)


def get_lora_layer_map(
    peft_model,
    target_modules: Optional[Sequence[str]] = None,
) -> Dict[str, torch.nn.Module]:
    """Map canonical base-weight names to LoRA-wrapped modules."""
    layer_map: Dict[str, torch.nn.Module] = {}

    for module_name, module in peft_model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "base_layer")
        ):
            continue
        if not hasattr(module.base_layer, "weight"):
            continue

        weight_name = canonicalize_weight_name(f"{module_name}.weight")
        if target_modules and not _weight_matches_targets(weight_name, target_modules):
            continue

        if weight_name in layer_map:
            logger.warning(
                "[FILA] Duplicate LoRA layer mapping for %s; keeping first occurrence.",
                weight_name,
            )
            continue
        layer_map[weight_name] = module

    return layer_map


def collect_fila_target_parameters(
    peft_model,
    target_modules: Optional[Sequence[str]] = None,
) -> Dict[str, torch.nn.Parameter]:
    """Collect canonical base-weight parameters used by FILA."""
    layer_map = get_lora_layer_map(peft_model=peft_model, target_modules=target_modules)
    return {name: layer.base_layer.weight for name, layer in layer_map.items()}


def _canonicalize_importance_dict(raw_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in raw_map.items():
        canon = canonicalize_weight_name(name)
        if not canon.endswith(".weight"):
            continue
        out[canon] = tensor
    return out


def _extract_scaling(layer, adapter_name: str) -> float:
    scaling_attr = getattr(layer, "scaling", None)
    if scaling_attr is None:
        raise ValueError("LoRA layer has no scaling attribute.")

    if isinstance(scaling_attr, dict):
        if adapter_name not in scaling_attr:
            raise ValueError(f"Adapter '{adapter_name}' not found in LoRA layer scaling map.")
        scaling = scaling_attr[adapter_name]
    else:
        scaling = scaling_attr

    if isinstance(scaling, torch.Tensor):
        scaling = float(scaling.detach().item())
    scaling = float(scaling)
    if scaling == 0.0:
        raise ValueError("LoRA scaling is zero; FILA initialization is undefined.")
    return scaling


@torch.no_grad()
def apply_fila_initialization(
    peft_model,
    importance_file: str,
    target_modules: Optional[Sequence[str]] = None,
    lora_rank: Optional[int] = None,
    eps: float = 1e-5,
    adapter_name: str = "default",
    strict: bool = True,
    run_sanity_check: bool = True,
):
    """Apply FILA initialization and residual rewrite to LoRA-wrapped layers."""
    if not os.path.exists(importance_file):
        raise FileNotFoundError(f"Importance file not found: {importance_file}")

    payload = torch.load(importance_file, map_location="cpu")
    required = {"importance_f", "importance_r", "f_cnt", "r_cnt"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(
            f"Importance file is missing required keys: {sorted(missing)}"
        )

    importance_f = _canonicalize_importance_dict(payload["importance_f"])
    importance_r = _canonicalize_importance_dict(payload["importance_r"])

    f_cnt = float(payload["f_cnt"])
    r_cnt = float(payload["r_cnt"])
    if f_cnt <= 0 or r_cnt <= 0:
        raise ValueError(
            f"Invalid token counters in importance file: f_cnt={f_cnt}, r_cnt={r_cnt}"
        )

    layer_map = get_lora_layer_map(peft_model=peft_model, target_modules=target_modules)
    if not layer_map:
        raise ValueError(
            "No LoRA-wrapped target layers found for FILA initialization."
        )

    matched = 0
    max_rel_error = 0.0
    skipped_missing_importance = []
    skipped_shape = []

    for weight_name, layer in layer_map.items():
        imp_f = importance_f.get(weight_name)
        imp_r = importance_r.get(weight_name)
        if imp_f is None or imp_r is None:
            skipped_missing_importance.append(weight_name)
            continue

        if adapter_name not in layer.lora_A or adapter_name not in layer.lora_B:
            raise ValueError(
                f"Adapter '{adapter_name}' not found in LoRA layer for {weight_name}."
            )

        base_weight = layer.base_layer.weight.data
        if tuple(imp_f.shape) != tuple(base_weight.shape) or tuple(imp_r.shape) != tuple(
            base_weight.shape
        ):
            skipped_shape.append(
                (
                    weight_name,
                    tuple(base_weight.shape),
                    tuple(imp_f.shape),
                    tuple(imp_r.shape),
                )
            )
            continue

        device = base_weight.device
        weight_fp32 = base_weight.float()
        imp = (imp_f.float() / f_cnt) / (float(eps) + (imp_r.float() / r_cnt))
        imp = imp.to(device=device, dtype=torch.float32)

        row_importance = imp.sum(dim=1).clamp_min(0.0).sqrt()
        weighted_w = row_importance.unsqueeze(1) * weight_fp32

        a_weight = layer.lora_A[adapter_name].weight.data
        b_weight = layer.lora_B[adapter_name].weight.data
        rank = int(lora_rank if lora_rank is not None else a_weight.shape[0])
        max_rank = min(
            weighted_w.shape[0],
            weighted_w.shape[1],
            rank,
            a_weight.shape[0],
            b_weight.shape[1],
        )
        if max_rank <= 0:
            skipped_shape.append(
                (
                    weight_name,
                    tuple(base_weight.shape),
                    tuple(imp_f.shape),
                    tuple(imp_r.shape),
                )
            )
            continue

        u, s, v = torch.svd_lowrank(weighted_w, q=max_rank)
        scaling = _extract_scaling(layer, adapter_name=adapter_name)

        # Official FILA correction divides singular values by LoRA scaling.
        s = (s / scaling).clamp_min(0.0)
        sqrt_s = torch.sqrt(s)

        new_a = (v * sqrt_s.unsqueeze(0)).t().contiguous()  # [r, in]
        new_b = (
            (u * sqrt_s.unsqueeze(0))
            / (row_importance.unsqueeze(1) + float(eps))
        ).contiguous()  # [out, r]

        a_weight.zero_()
        b_weight.zero_()
        a_weight[:max_rank, :].copy_(new_a.to(device=a_weight.device, dtype=a_weight.dtype))
        b_weight[:, :max_rank].copy_(new_b.to(device=b_weight.device, dtype=b_weight.dtype))

        original_w = weight_fp32
        delta = (b_weight @ a_weight).to(device=device, dtype=torch.float32)
        residual = original_w - scaling * delta
        base_weight.copy_(residual.to(dtype=base_weight.dtype))

        if run_sanity_check:
            recon = base_weight.float() + scaling * (
                layer.lora_B[adapter_name].weight.data.float()
                @ layer.lora_A[adapter_name].weight.data.float()
            )
            rel_error = torch.norm(recon - original_w) / (torch.norm(original_w) + 1e-12)
            max_rel_error = max(max_rel_error, float(rel_error.item()))

        matched += 1

    if strict and matched == 0:
        raise ValueError(
            "FILA did not match any layers. Check target_modules, naming, and importance file contents."
        )

    if skipped_shape:
        logger.warning("[FILA] Skipped %d layers due to shape mismatch.", len(skipped_shape))
    if skipped_missing_importance:
        logger.warning(
            "[FILA] Skipped %d layers due to missing importance entries.",
            len(skipped_missing_importance),
        )

    stats = {
        "matched_layers": matched,
        "available_lora_layers": len(layer_map),
        "missing_importance_layers": len(skipped_missing_importance),
        "shape_mismatch_layers": len(skipped_shape),
        "max_reconstruction_rel_error": max_rel_error,
    }
    logger.info("[FILA] Initialization stats: %s", stats)
    return stats
