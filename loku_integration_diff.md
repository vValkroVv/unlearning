# LoKU Integration Diff

Base commit: 8f1713b14607d44d82b9b4b5b4c54e30ffc0b3c8 (before LoKU integration)
Target: current working tree

```diff
diff --git a/prod-gpu-runs.md b/prod-gpu-runs.md
index 0b7c5f3..673e1dc 100644
--- a/prod-gpu-runs.md
+++ b/prod-gpu-runs.md
@@ -204,3 +204,45 @@ EVAL_BATCH_SIZE=8 \
 DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
 bash scripts/popqa/npo_sam_popqa.sh
 ```
+
+## 9) LoKU - DUET
+
+```bash
+CUDA_DEVICE_ORDER=PCI_BUS_ID \
+CUDA_VISIBLE_DEVICES=1 \
+USE_SFT_BASE=1 \
+LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
+SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
+MERGE_POPULARITY_FORGET=1 \
+PER_DEVICE_TRAIN_BS=1 \
+GRAD_ACCUM=32 \
+IMPORTANCE_BATCH_SIZE=1 \
+IMPORTANCE_MAX_STEPS=0 \
+EVAL_BATCH_SIZE=8 \
+DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
+LRS="1e-4" \
+bash scripts/duet/loku_duet.sh
+```
+
+## 10) LoKU - UNLamb
+
+```bash
+CUDA_DEVICE_ORDER=PCI_BUS_ID \
+CUDA_VISIBLE_DEVICES=1 \
+USE_SFT_BASE=1 \
+LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
+SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
+MERGE_POPULARITY_FORGET=1 \
+PER_DEVICE_TRAIN_BS=1 \
+GRAD_ACCUM=32 \
+IMPORTANCE_BATCH_SIZE=1 \
+IMPORTANCE_MAX_STEPS=0 \
+EVAL_BATCH_SIZE=8 \
+DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
+LRS="1e-4" \
+bash scripts/popqa/loku_popqa.sh
+```
+
+Notes:
+- LoKU includes an extra importance-measurement stage before training, so keep `IMPORTANCE_BATCH_SIZE` conservative.
+- For smoke checks use `IMPORTANCE_MAX_STEPS` (for example `50`) before full runs.
diff --git a/prod-runs.md b/prod-runs.md
index d83431a..dcf69cb 100644
--- a/prod-runs.md
+++ b/prod-runs.md
@@ -183,3 +183,43 @@ GRAD_ACCUM=32 \
 EVAL_BATCH_SIZE=8 \
 bash scripts/popqa/npo_sam_popqa.sh
 ```
+
+## 9) LoKU - DUET
+
+```bash
+CUDA_DEVICE_ORDER=PCI_BUS_ID \
+CUDA_VISIBLE_DEVICES=1 \
+USE_SFT_BASE=1 \
+LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
+SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
+MERGE_POPULARITY_FORGET=1 \
+PER_DEVICE_TRAIN_BS=1 \
+GRAD_ACCUM=32 \
+IMPORTANCE_BATCH_SIZE=1 \
+IMPORTANCE_MAX_STEPS=0 \
+EVAL_BATCH_SIZE=8 \
+LRS="1e-4" \
+bash scripts/duet/loku_duet.sh
+```
+
+## 10) LoKU - UNLamb
+
+```bash
+CUDA_DEVICE_ORDER=PCI_BUS_ID \
+CUDA_VISIBLE_DEVICES=1 \
+USE_SFT_BASE=1 \
+LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
+SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
+MERGE_POPULARITY_FORGET=1 \
+PER_DEVICE_TRAIN_BS=1 \
+GRAD_ACCUM=32 \
+IMPORTANCE_BATCH_SIZE=1 \
+IMPORTANCE_MAX_STEPS=0 \
+EVAL_BATCH_SIZE=8 \
+LRS="1e-4" \
+bash scripts/popqa/loku_popqa.sh
+```
+
+Notes:
+- LoKU runs a separate importance pass before training; keep `IMPORTANCE_BATCH_SIZE` small (usually `1`) to avoid OOM.
+- If you only need a quick validation run, set `IMPORTANCE_MAX_STEPS` to a small value (for example `50`).
diff --git a/src/trainer/__init__.py b/src/trainer/__init__.py
index 67edfff..c43a2d5 100644
--- a/src/trainer/__init__.py
+++ b/src/trainer/__init__.py
@@ -21,6 +21,7 @@ from trainer.unlearn.ada_pop import AdaPop
 from trainer.unlearn.pop_dynam_b_wga import PopDynamBWGA
 from trainer.unlearn.falcon import FALCON
 from trainer.unlearn.r2d import R2D
+from trainer.unlearn.loku import LoKU
 
 
 import logging
@@ -111,3 +112,4 @@ _register_trainer(AdaPop)
 _register_trainer(PopDynamBWGA)
 _register_trainer(FALCON)
 _register_trainer(R2D)
+_register_trainer(LoKU)
diff --git a/src/trainer/utils.py b/src/trainer/utils.py
index 3bd2212..a74c45a 100644
--- a/src/trainer/utils.py
+++ b/src/trainer/utils.py
@@ -144,6 +144,39 @@ def compute_wga_loss(model, inputs, beta):
     return forget_loss, outputs
 
 
+def ihl_loss_from_logits(
+    logits: torch.Tensor,
+    labels: torch.Tensor,
+    ignore_index: int = -100,
+    alpha: float = 1.0,
+) -> torch.Tensor:
+    """Inverted Hinge Loss (IHL) on next-token probabilities.
+
+    Minimizing this loss encourages the true-token probability to be lower than
+    the strongest alternative by a margin alpha.
+    """
+    shift_logits = logits[..., :-1, :].contiguous()
+    shift_labels = labels[..., 1:].contiguous()
+
+    valid_mask = shift_labels != ignore_index
+    if not valid_mask.any():
+        return shift_logits.new_zeros(())
+
+    probs = shift_logits.softmax(dim=-1)
+    probs = probs[valid_mask]  # [N, V]
+    targets = shift_labels[valid_mask]  # [N]
+
+    p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
+    probs_other = probs.clone()
+    probs_other.scatter_(1, targets.unsqueeze(1), float("-inf"))
+    p_other = probs_other.max(dim=1).values  # [N]
+    margins = p_true - p_other
+
+    # Inverted hinge: alpha + margin (official LoKU convention).
+    measures = (float(alpha) + margins).clamp_min(0.0)
+    return measures.mean()
+
+
 def beta_from_pop_sum_tensor(pop_sum: torch.Tensor) -> torch.Tensor:
     """Compute dynamic beta from a tensor of pop_sum using the clipped power-law.
 
diff --git a/configs/trainer/LoKU.yaml b/configs/trainer/LoKU.yaml
new file mode 100644
index 0000000..a5df4bd
--- /dev/null
+++ b/configs/trainer/LoKU.yaml
@@ -0,0 +1,20 @@
+defaults:
+  - GradDiff
+
+handler: LoKU
+
+args:
+  learning_rate: 1e-4
+  num_train_epochs: 5
+
+method_args:
+  gamma: 1.0
+  alpha: 1.0
+  ihl_alpha: 1.0
+  retain_loss_type: NLL
+
+  importance_file: null
+  fila_eps: 1e-5
+  fila_adapter_name: default
+  fila_base_subdir: base_model
+  run_fila_sanity_check: true
diff --git a/configs/experiment/unlearn/duet/loku_lora.yaml b/configs/experiment/unlearn/duet/loku_lora.yaml
new file mode 100644
index 0000000..40afafa
--- /dev/null
+++ b/configs/experiment/unlearn/duet/loku_lora.yaml
@@ -0,0 +1,71 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: LoKU
+  - override /data: unlearn
+  - override /data/datasets@data.forget: DUET_QA_forget
+  - override /data/datasets@data.retain: DUET_QA_retain
+  - override /eval: duet
+
+forget_split: city_forget_rare_5
+retain_split: city_fast_retain_500
+holdout_split: ${retain_split}
+retain_logs_path: null
+question_key: question
+
+model:
+  model_args:
+    pretrained_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
+
+data:
+  anchor: forget
+  forget:
+    DUET_QA_forget:
+      args:
+        hf_args:
+          path: SwetieePawsss/DUET
+          split: ${forget_split}
+        question_key: ${question_key}
+  retain:
+    DUET_QA_retain:
+      args:
+        hf_args:
+          path: SwetieePawsss/DUET
+          split: ${retain_split}
+        question_key: ${question_key}
+
+eval:
+  duet:
+    forget_split: ${forget_split}
+    holdout_split: ${holdout_split}
+    retain_logs_path: ${retain_logs_path}
+    question_key: ${question_key}
+
+trainer:
+  args:
+    per_device_train_batch_size: 1
+    gradient_accumulation_steps: 32
+    learning_rate: 1e-4
+    num_train_epochs: 5
+    lr_scheduler_type: constant
+    warmup_ratio: 0.1
+    logging_steps: 10
+    eval_strategy: "no"
+    save_strategy: "no"
+    do_eval: false
+    eval_on_start: false
+    remove_unused_columns: false
+    gradient_checkpointing: false
+    ddp_find_unused_parameters: false
+  method_args:
+    ihl_alpha: 1.0
+    alpha: 1.0
+    gamma: 1.0
+    retain_loss_type: NLL
+    importance_file: null
+    fila_eps: 1e-5
+    fila_adapter_name: default
+    fila_base_subdir: base_model
+
+task_name: duet_loku_lora
diff --git a/configs/experiment/unlearn/popqa/loku_lora.yaml b/configs/experiment/unlearn/popqa/loku_lora.yaml
new file mode 100644
index 0000000..b54c315
--- /dev/null
+++ b/configs/experiment/unlearn/popqa/loku_lora.yaml
@@ -0,0 +1,71 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: LoKU
+  - override /data: unlearn
+  - override /data/datasets@data.forget: POPQA_QA_forget
+  - override /data/datasets@data.retain: POPQA_QA_retain
+  - override /eval: popqa
+
+forget_split: rare_forget5_sum
+retain_split: fast_retain_500
+holdout_split: ${retain_split}
+retain_logs_path: null
+question_key: question
+
+model:
+  model_args:
+    pretrained_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
+
+data:
+  anchor: forget
+  forget:
+    POPQA_QA_forget:
+      args:
+        hf_args:
+          path: SwetieePawsss/exp_UNLamb
+          split: ${forget_split}
+        question_key: ${question_key}
+  retain:
+    POPQA_QA_retain:
+      args:
+        hf_args:
+          path: SwetieePawsss/exp_UNLamb
+          split: ${retain_split}
+        question_key: ${question_key}
+
+eval:
+  duet:
+    forget_split: ${forget_split}
+    holdout_split: ${holdout_split}
+    retain_logs_path: ${retain_logs_path}
+    question_key: ${question_key}
+
+trainer:
+  args:
+    per_device_train_batch_size: 1
+    gradient_accumulation_steps: 32
+    learning_rate: 1e-4
+    num_train_epochs: 5
+    lr_scheduler_type: constant
+    warmup_ratio: 0.1
+    logging_steps: 10
+    eval_strategy: "no"
+    save_strategy: "no"
+    do_eval: false
+    eval_on_start: false
+    remove_unused_columns: false
+    gradient_checkpointing: false
+    ddp_find_unused_parameters: false
+  method_args:
+    ihl_alpha: 1.0
+    alpha: 1.0
+    gamma: 1.0
+    retain_loss_type: NLL
+    importance_file: null
+    fila_eps: 1e-5
+    fila_adapter_name: default
+    fila_base_subdir: base_model
+
+task_name: popqa_loku_lora
diff --git a/src/model/fila.py b/src/model/fila.py
new file mode 100644
index 0000000..064839d
--- /dev/null
+++ b/src/model/fila.py
@@ -0,0 +1,249 @@
+import logging
+import os
+from typing import Dict, Optional, Sequence
+
+import torch
+
+logger = logging.getLogger(__name__)
+
+
+def canonicalize_weight_name(name: str) -> str:
+    """Normalize parameter/module names to a stable base-weight identifier."""
+    normalized = str(name)
+    while normalized.startswith("module."):
+        normalized = normalized[len("module.") :]
+    while normalized.startswith("base_model.model."):
+        normalized = normalized[len("base_model.model.") :]
+    normalized = normalized.replace(".base_layer.weight", ".weight")
+    return normalized
+
+
+def _weight_matches_targets(weight_name: str, target_modules: Sequence[str]) -> bool:
+    return any(weight_name.endswith(f".{target}.weight") for target in target_modules)
+
+
+def get_lora_layer_map(
+    peft_model,
+    target_modules: Optional[Sequence[str]] = None,
+) -> Dict[str, torch.nn.Module]:
+    """Map canonical base-weight names to LoRA-wrapped modules."""
+    layer_map: Dict[str, torch.nn.Module] = {}
+
+    for module_name, module in peft_model.named_modules():
+        if not (
+            hasattr(module, "lora_A")
+            and hasattr(module, "lora_B")
+            and hasattr(module, "base_layer")
+        ):
+            continue
+        if not hasattr(module.base_layer, "weight"):
+            continue
+
+        weight_name = canonicalize_weight_name(f"{module_name}.weight")
+        if target_modules and not _weight_matches_targets(weight_name, target_modules):
+            continue
+
+        if weight_name in layer_map:
+            logger.warning(
+                "[FILA] Duplicate LoRA layer mapping for %s; keeping first occurrence.",
+                weight_name,
+            )
+            continue
+        layer_map[weight_name] = module
+
+    return layer_map
+
+
+def collect_fila_target_parameters(
+    peft_model,
+    target_modules: Optional[Sequence[str]] = None,
+) -> Dict[str, torch.nn.Parameter]:
+    """Collect canonical base-weight parameters used by FILA."""
+    layer_map = get_lora_layer_map(peft_model=peft_model, target_modules=target_modules)
+    return {name: layer.base_layer.weight for name, layer in layer_map.items()}
+
+
+def _canonicalize_importance_dict(raw_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
+    out: Dict[str, torch.Tensor] = {}
+    for name, tensor in raw_map.items():
+        canon = canonicalize_weight_name(name)
+        if not canon.endswith(".weight"):
+            continue
+        out[canon] = tensor
+    return out
+
+
+def _extract_scaling(layer, adapter_name: str) -> float:
+    scaling_attr = getattr(layer, "scaling", None)
+    if scaling_attr is None:
+        raise ValueError("LoRA layer has no scaling attribute.")
+
+    if isinstance(scaling_attr, dict):
+        if adapter_name not in scaling_attr:
+            raise ValueError(f"Adapter '{adapter_name}' not found in LoRA layer scaling map.")
+        scaling = scaling_attr[adapter_name]
+    else:
+        scaling = scaling_attr
+
+    if isinstance(scaling, torch.Tensor):
+        scaling = float(scaling.detach().item())
+    scaling = float(scaling)
+    if scaling == 0.0:
+        raise ValueError("LoRA scaling is zero; FILA initialization is undefined.")
+    return scaling
+
+
+@torch.no_grad()
+def apply_fila_initialization(
+    peft_model,
+    importance_file: str,
+    target_modules: Optional[Sequence[str]] = None,
+    lora_rank: Optional[int] = None,
+    eps: float = 1e-5,
+    adapter_name: str = "default",
+    strict: bool = True,
+    run_sanity_check: bool = True,
+):
+    """Apply FILA initialization and residual rewrite to LoRA-wrapped layers."""
+    if not os.path.exists(importance_file):
+        raise FileNotFoundError(f"Importance file not found: {importance_file}")
+
+    payload = torch.load(importance_file, map_location="cpu")
+    required = {"importance_f", "importance_r", "f_cnt", "r_cnt"}
+    missing = required.difference(payload.keys())
+    if missing:
+        raise ValueError(
+            f"Importance file is missing required keys: {sorted(missing)}"
+        )
+
+    importance_f = _canonicalize_importance_dict(payload["importance_f"])
+    importance_r = _canonicalize_importance_dict(payload["importance_r"])
+
+    f_cnt = float(payload["f_cnt"])
+    r_cnt = float(payload["r_cnt"])
+    if f_cnt <= 0 or r_cnt <= 0:
+        raise ValueError(
+            f"Invalid token counters in importance file: f_cnt={f_cnt}, r_cnt={r_cnt}"
+        )
+
+    layer_map = get_lora_layer_map(peft_model=peft_model, target_modules=target_modules)
+    if not layer_map:
+        raise ValueError(
+            "No LoRA-wrapped target layers found for FILA initialization."
+        )
+
+    matched = 0
+    max_rel_error = 0.0
+    skipped_missing_importance = []
+    skipped_shape = []
+
+    for weight_name, layer in layer_map.items():
+        imp_f = importance_f.get(weight_name)
+        imp_r = importance_r.get(weight_name)
+        if imp_f is None or imp_r is None:
+            skipped_missing_importance.append(weight_name)
+            continue
+
+        if adapter_name not in layer.lora_A or adapter_name not in layer.lora_B:
+            raise ValueError(
+                f"Adapter '{adapter_name}' not found in LoRA layer for {weight_name}."
+            )
+
+        base_weight = layer.base_layer.weight.data
+        if tuple(imp_f.shape) != tuple(base_weight.shape) or tuple(imp_r.shape) != tuple(
+            base_weight.shape
+        ):
+            skipped_shape.append(
+                (
+                    weight_name,
+                    tuple(base_weight.shape),
+                    tuple(imp_f.shape),
+                    tuple(imp_r.shape),
+                )
+            )
+            continue
+
+        device = base_weight.device
+        weight_fp32 = base_weight.float()
+        imp = (imp_f.float() / f_cnt) / (float(eps) + (imp_r.float() / r_cnt))
+        imp = imp.to(device=device, dtype=torch.float32)
+
+        row_importance = imp.sum(dim=1).clamp_min(0.0).sqrt()
+        weighted_w = row_importance.unsqueeze(1) * weight_fp32
+
+        a_weight = layer.lora_A[adapter_name].weight.data
+        b_weight = layer.lora_B[adapter_name].weight.data
+        rank = int(lora_rank if lora_rank is not None else a_weight.shape[0])
+        max_rank = min(
+            weighted_w.shape[0],
+            weighted_w.shape[1],
+            rank,
+            a_weight.shape[0],
+            b_weight.shape[1],
+        )
+        if max_rank <= 0:
+            skipped_shape.append(
+                (
+                    weight_name,
+                    tuple(base_weight.shape),
+                    tuple(imp_f.shape),
+                    tuple(imp_r.shape),
+                )
+            )
+            continue
+
+        u, s, v = torch.svd_lowrank(weighted_w, q=max_rank)
+        scaling = _extract_scaling(layer, adapter_name=adapter_name)
+
+        # Official FILA correction divides singular values by LoRA scaling.
+        s = (s / scaling).clamp_min(0.0)
+        sqrt_s = torch.sqrt(s)
+
+        new_a = (v * sqrt_s.unsqueeze(0)).t().contiguous()  # [r, in]
+        new_b = (
+            (u * sqrt_s.unsqueeze(0))
+            / (row_importance.unsqueeze(1) + float(eps))
+        ).contiguous()  # [out, r]
+
+        a_weight.zero_()
+        b_weight.zero_()
+        a_weight[:max_rank, :].copy_(new_a.to(device=a_weight.device, dtype=a_weight.dtype))
+        b_weight[:, :max_rank].copy_(new_b.to(device=b_weight.device, dtype=b_weight.dtype))
+
+        original_w = weight_fp32
+        delta = (b_weight @ a_weight).to(device=device, dtype=torch.float32)
+        residual = original_w - scaling * delta
+        base_weight.copy_(residual.to(dtype=base_weight.dtype))
+
+        if run_sanity_check:
+            recon = base_weight.float() + scaling * (
+                layer.lora_B[adapter_name].weight.data.float()
+                @ layer.lora_A[adapter_name].weight.data.float()
+            )
+            rel_error = torch.norm(recon - original_w) / (torch.norm(original_w) + 1e-12)
+            max_rel_error = max(max_rel_error, float(rel_error.item()))
+
+        matched += 1
+
+    if strict and matched == 0:
+        raise ValueError(
+            "FILA did not match any layers. Check target_modules, naming, and importance file contents."
+        )
+
+    if skipped_shape:
+        logger.warning("[FILA] Skipped %d layers due to shape mismatch.", len(skipped_shape))
+    if skipped_missing_importance:
+        logger.warning(
+            "[FILA] Skipped %d layers due to missing importance entries.",
+            len(skipped_missing_importance),
+        )
+
+    stats = {
+        "matched_layers": matched,
+        "available_lora_layers": len(layer_map),
+        "missing_importance_layers": len(skipped_missing_importance),
+        "shape_mismatch_layers": len(skipped_shape),
+        "max_reconstruction_rel_error": max_rel_error,
+    }
+    logger.info("[FILA] Initialization stats: %s", stats)
+    return stats
diff --git a/src/trainer/unlearn/loku.py b/src/trainer/unlearn/loku.py
new file mode 100644
index 0000000..81a02d9
--- /dev/null
+++ b/src/trainer/unlearn/loku.py
@@ -0,0 +1,167 @@
+import logging
+import os
+from typing import Optional
+
+from model.fila import apply_fila_initialization
+from trainer.unlearn.grad_diff import GradDiff
+from trainer.utils import ihl_loss_from_logits
+
+logger = logging.getLogger(__name__)
+
+
+class LoKU(GradDiff):
+    """LoKU: Inverted Hinge forget objective + FILA initialization for LoRA."""
+
+    def __init__(
+        self,
+        ihl_alpha: float = 1.0,
+        importance_file: Optional[str] = None,
+        fila_eps: float = 1e-5,
+        fila_adapter_name: str = "default",
+        fila_base_subdir: str = "base_model",
+        run_fila_sanity_check: bool = True,
+        *args,
+        **kwargs,
+    ):
+        super().__init__(*args, **kwargs)
+        self.ihl_alpha = float(ihl_alpha)
+        self.importance_file = importance_file
+        self.fila_eps = float(fila_eps)
+        self.fila_adapter_name = str(fila_adapter_name)
+        self.fila_base_subdir = str(fila_base_subdir)
+        self.run_fila_sanity_check = bool(run_fila_sanity_check)
+
+        self._fila_enabled = False
+        self._fila_applied = False
+        self._fila_stats = None
+        self._fila_base_saved = False
+
+        if self.importance_file not in (None, "", "null", "None"):
+            self._fila_enabled = True
+            self._apply_fila_or_raise()
+
+    def _as_model_inputs(self, batch):
+        if isinstance(batch, dict) and "original" in batch:
+            batch = batch["original"]
+        return {
+            "input_ids": batch["input_ids"],
+            "attention_mask": batch["attention_mask"],
+            "labels": batch["labels"],
+        }
+
+    def _resolve_lora_targets_and_rank(self):
+        peft_config = getattr(self.model, "peft_config", None)
+        if peft_config is None:
+            raise ValueError(
+                "LoKU with FILA requires a LoRA/PEFT model, but no peft_config was found."
+            )
+
+        if not isinstance(peft_config, dict) or not peft_config:
+            raise ValueError("Invalid peft_config structure on model.")
+
+        cfg = peft_config.get(self.fila_adapter_name)
+        if cfg is None:
+            cfg = next(iter(peft_config.values()))
+            logger.warning(
+                "[LoKU] Adapter '%s' not found; using first available adapter config.",
+                self.fila_adapter_name,
+            )
+
+        target_modules = list(getattr(cfg, "target_modules", []) or [])
+        if not target_modules:
+            raise ValueError("LoKU FILA needs non-empty model.lora_config.target_modules.")
+
+        rank = int(getattr(cfg, "r", 0))
+        if rank <= 0:
+            raise ValueError("LoKU FILA requires LoRA rank `r` > 0.")
+
+        return target_modules, rank
+
+    def _apply_fila_or_raise(self):
+        importance_file = str(self.importance_file)
+        if not os.path.exists(importance_file):
+            raise FileNotFoundError(
+                f"[LoKU] importance_file does not exist: {importance_file}"
+            )
+
+        target_modules, lora_rank = self._resolve_lora_targets_and_rank()
+        logger.info(
+            "[LoKU] Applying FILA from %s (targets=%s, rank=%d, eps=%g, adapter=%s)",
+            importance_file,
+            target_modules,
+            lora_rank,
+            self.fila_eps,
+            self.fila_adapter_name,
+        )
+        self._fila_stats = apply_fila_initialization(
+            peft_model=self.model,
+            importance_file=importance_file,
+            target_modules=target_modules,
+            lora_rank=lora_rank,
+            eps=self.fila_eps,
+            adapter_name=self.fila_adapter_name,
+            strict=True,
+            run_sanity_check=self.run_fila_sanity_check,
+        )
+        self._fila_applied = True
+
+    def compute_loss(self, model, inputs, return_outputs=False):
+        forget_inputs = self._as_model_inputs(inputs["forget"])
+        forget_outputs = model(**forget_inputs)
+        forget_loss = ihl_loss_from_logits(
+            logits=forget_outputs.logits,
+            labels=forget_inputs["labels"],
+            alpha=self.ihl_alpha,
+        )
+
+        retain_inputs = self._as_model_inputs(inputs["retain"])
+        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
+
+        loss = self.gamma * forget_loss + self.alpha * retain_loss
+        return (loss, forget_outputs) if return_outputs else loss
+
+    def _save_fila_base_model(self, output_dir: str) -> None:
+        if self._fila_base_saved:
+            return
+        if not self.is_world_process_zero():
+            return
+
+        base_dir = os.path.join(output_dir, self.fila_base_subdir)
+        os.makedirs(base_dir, exist_ok=True)
+
+        model_to_save = self.model
+        if getattr(self, "accelerator", None) is not None:
+            try:
+                model_to_save = self.accelerator.unwrap_model(self.model)
+            except Exception:
+                model_to_save = self.model
+
+        if hasattr(model_to_save, "unload") and callable(model_to_save.unload):
+            base_model = model_to_save.unload()
+        else:
+            raise ValueError(
+                "[LoKU] FILA base save requires a PEFT model with `unload()` support."
+            )
+
+        base_model.save_pretrained(base_dir)
+        if self.tokenizer is not None:
+            self.tokenizer.save_pretrained(base_dir)
+
+        self._fila_base_saved = True
+        logger.info("[LoKU] Saved FILA residual base model to %s", base_dir)
+
+    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
+        out_dir = output_dir or self.args.output_dir
+        result = super().save_model(output_dir=output_dir, _internal_call=_internal_call)
+
+        if not self._fila_enabled:
+            return result
+        if not self._fila_applied:
+            raise ValueError("[LoKU] FILA is enabled but was not successfully applied.")
+        if _internal_call:
+            return result
+        if "checkpoint-" in str(out_dir):
+            return result
+
+        self._save_fila_base_model(output_dir=out_dir)
+        return result
diff --git a/src/tools/loku_measure_importance.py b/src/tools/loku_measure_importance.py
new file mode 100755
index 0000000..4cb0dd7
--- /dev/null
+++ b/src/tools/loku_measure_importance.py
@@ -0,0 +1,277 @@
+#!/usr/bin/env python3
+"""Measure LoKU FILA importance tensors on forget/retain batches.
+
+Produces an artifact with keys:
+- importance_f: Dict[name -> Tensor]
+- importance_r: Dict[name -> Tensor]
+- f_cnt: int
+- r_cnt: int
+- target_modules: List[str]
+- meta: Dict
+"""
+
+from __future__ import annotations
+
+import argparse
+import os
+import sys
+import time
+from pathlib import Path
+from typing import Dict, List, Tuple
+
+import torch
+from hydra import compose, initialize_config_dir
+from omegaconf import open_dict
+from torch.utils.data import DataLoader
+
+SRC_ROOT = Path(__file__).resolve().parent.parent
+if str(SRC_ROOT) not in sys.path:
+    sys.path.insert(0, str(SRC_ROOT))
+
+from data import get_collators, get_data
+from data.utils import IGNORE_INDEX
+from model import get_model
+from model.fila import collect_fila_target_parameters
+from trainer.utils import _filter_model_inputs, seed_everything
+
+
+def _log(msg: str, quiet: bool) -> None:
+    if quiet:
+        return
+    ts = time.strftime("%Y-%m-%d %H:%M:%S")
+    print(f"[LoKU-IMP][{ts}] {msg}", flush=True)
+
+
+def _parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Measure LoKU FILA importance tensors.")
+    parser.add_argument("--config-name", default="unlearn.yaml")
+    parser.add_argument("--experiment", required=True)
+    parser.add_argument("--output-path", required=True)
+    parser.add_argument("--max-steps", type=int, default=0)
+    parser.add_argument("--batch-size", type=int, default=None)
+    parser.add_argument("--num-workers", type=int, default=0)
+    parser.add_argument("--seed", type=int, default=0)
+    parser.add_argument("--device", default=None)
+    parser.add_argument("--quiet", action="store_true")
+    parser.add_argument(
+        "overrides",
+        nargs=argparse.REMAINDER,
+        help="Extra Hydra overrides. Prefix with '--' before first override.",
+    )
+    return parser.parse_args()
+
+
+def _compose_cfg(args: argparse.Namespace):
+    overrides = [f"experiment={args.experiment}"]
+    extra = list(args.overrides)
+    if extra and extra[0] == "--":
+        extra = extra[1:]
+    overrides.extend(extra)
+
+    config_dir = str(SRC_ROOT.parent / "configs")
+    with initialize_config_dir(version_base=None, config_dir=config_dir):
+        cfg = compose(config_name=args.config_name, overrides=overrides)
+    return cfg, extra
+
+
+def _resolve_target_modules(cfg) -> List[str]:
+    lora_cfg = cfg.model.get("lora_config", None)
+    if lora_cfg is None:
+        raise ValueError("LoKU importance measurement requires a LoRA model config.")
+    target_modules = list(lora_cfg.get("target_modules", []) or [])
+    if not target_modules:
+        raise ValueError("model.lora_config.target_modules must be non-empty.")
+    return target_modules
+
+
+def _select_device(args: argparse.Namespace) -> str:
+    if args.device:
+        return str(args.device)
+    return "cuda" if torch.cuda.is_available() else "cpu"
+
+
+def _prepare_model_and_data(cfg, args: argparse.Namespace):
+    model_cfg = cfg.model
+    template_args = model_cfg.template_args
+
+    model, tokenizer = get_model(model_cfg)
+    device = _select_device(args)
+    if getattr(model, "hf_device_map", None) is None:
+        model = model.to(device)
+
+    mode = cfg.get("mode", "unlearn")
+    data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args)
+    if "train" not in data:
+        raise ValueError("Expected `train` split in unlearn data pipeline.")
+
+    collator = get_collators(cfg.collator, tokenizer=tokenizer)
+    bs = args.batch_size
+    if bs is None:
+        bs = int(cfg.trainer.args.get("per_device_train_batch_size", 1))
+
+    dataloader = DataLoader(
+        data["train"],
+        batch_size=max(1, int(bs)),
+        shuffle=False,
+        num_workers=max(0, int(args.num_workers)),
+        collate_fn=collator,
+    )
+    return model, tokenizer, dataloader, device
+
+
+def _to_model_inputs(batch_split, device: str) -> Dict[str, torch.Tensor]:
+    if isinstance(batch_split, dict) and "original" in batch_split:
+        batch_split = batch_split["original"]
+    model_inputs = _filter_model_inputs(batch_split)
+    if "labels" not in model_inputs:
+        raise ValueError("Batch is missing labels required for CE-based importance.")
+    return {
+        key: value.to(device) if hasattr(value, "to") else value
+        for key, value in model_inputs.items()
+    }
+
+
+def _accumulate_for_split(
+    model,
+    model_inputs: Dict[str, torch.Tensor],
+    target_params: Dict[str, torch.nn.Parameter],
+    accumulator: Dict[str, torch.Tensor],
+) -> int:
+    labels = model_inputs["labels"]
+    token_count = int((labels != IGNORE_INDEX).sum().item())
+    if token_count <= 0:
+        return 0
+
+    model.zero_grad(set_to_none=True)
+    outputs = model(**model_inputs)
+    outputs.loss.backward()
+
+    scale = float(token_count)
+    for name, param in target_params.items():
+        grad = param.grad
+        if grad is None:
+            continue
+        accumulator[name] += (grad.detach().float().pow(2) * scale).cpu()
+
+    model.zero_grad(set_to_none=True)
+    return token_count
+
+
+def _init_accumulators(
+    target_params: Dict[str, torch.nn.Parameter],
+) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
+    importance_f = {
+        name: torch.zeros_like(param.detach().float(), device="cpu")
+        for name, param in target_params.items()
+    }
+    importance_r = {
+        name: torch.zeros_like(param.detach().float(), device="cpu")
+        for name, param in target_params.items()
+    }
+    return importance_f, importance_r
+
+
+def main() -> None:
+    args = _parse_args()
+    seed_everything(args.seed)
+
+    cfg, user_overrides = _compose_cfg(args)
+
+    # Optional CLI batch-size override without requiring a Hydra override.
+    if args.batch_size is not None:
+        with open_dict(cfg):
+            cfg.trainer.args.per_device_train_batch_size = int(args.batch_size)
+
+    target_modules = _resolve_target_modules(cfg)
+    _log(f"Target modules: {target_modules}", quiet=args.quiet)
+
+    model, _, dataloader, device = _prepare_model_and_data(cfg, args)
+
+    target_params = collect_fila_target_parameters(
+        peft_model=model,
+        target_modules=target_modules,
+    )
+    if not target_params:
+        raise ValueError(
+            "No LoRA base-layer target weights found. Ensure model is LoRA-wrapped and "
+            "target_modules match injected adapters."
+        )
+
+    # For importance measurement we only need gradients on FILA-target base weights.
+    for param in model.parameters():
+        param.requires_grad_(False)
+    for param in target_params.values():
+        param.requires_grad_(True)
+
+    model.train()
+
+    importance_f, importance_r = _init_accumulators(target_params)
+    f_cnt = 0
+    r_cnt = 0
+    steps = 0
+
+    max_steps = int(args.max_steps)
+    for batch in dataloader:
+        if max_steps > 0 and steps >= max_steps:
+            break
+
+        if "forget" not in batch or "retain" not in batch:
+            raise ValueError("Expected batch with `forget` and `retain` keys.")
+
+        forget_inputs = _to_model_inputs(batch["forget"], device=device)
+        retain_inputs = _to_model_inputs(batch["retain"], device=device)
+
+        f_cnt += _accumulate_for_split(
+            model=model,
+            model_inputs=forget_inputs,
+            target_params=target_params,
+            accumulator=importance_f,
+        )
+        r_cnt += _accumulate_for_split(
+            model=model,
+            model_inputs=retain_inputs,
+            target_params=target_params,
+            accumulator=importance_r,
+        )
+
+        steps += 1
+        if not args.quiet and steps % 10 == 0:
+            _log(
+                f"Processed {steps} steps (f_cnt={f_cnt}, r_cnt={r_cnt}).",
+                quiet=False,
+            )
+
+    if f_cnt <= 0 or r_cnt <= 0:
+        raise ValueError(
+            f"Invalid counters after measurement: f_cnt={f_cnt}, r_cnt={r_cnt}."
+        )
+
+    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
+    payload = {
+        "importance_f": importance_f,
+        "importance_r": importance_r,
+        "f_cnt": int(f_cnt),
+        "r_cnt": int(r_cnt),
+        "target_modules": list(target_modules),
+        "meta": {
+            "config_name": args.config_name,
+            "experiment": args.experiment,
+            "overrides": user_overrides,
+            "steps": int(steps),
+            "max_steps": int(max_steps),
+            "batch_size": int(cfg.trainer.args.per_device_train_batch_size),
+            "seed": int(args.seed),
+            "device": device,
+        },
+    }
+    torch.save(payload, args.output_path)
+
+    _log(
+        f"Saved importance file to {args.output_path} "
+        f"(layers={len(importance_f)}, f_cnt={f_cnt}, r_cnt={r_cnt}, steps={steps})",
+        quiet=args.quiet,
+    )
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/duet/loku_duet.sh b/scripts/duet/loku_duet.sh
new file mode 100755
index 0000000..7665bbf
--- /dev/null
+++ b/scripts/duet/loku_duet.sh
@@ -0,0 +1,236 @@
+#!/bin/bash
+
+set -euo pipefail
+
+script_dir=$(dirname "$(realpath "$0")")
+repo_root=$(realpath "${script_dir}/../..")
+source "${script_dir}/_splits.sh"
+
+export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
+echo "Master Port: $MASTER_PORT"
+
+base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
+lora_model="${MODEL_CONFIG:-${base_model}-lora}"
+hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
+local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"
+sft_subfolder="${SFT_SUBFOLDER:-}"
+
+use_sft_base=${USE_SFT_BASE:-1}
+if [[ "${use_sft_base}" == "1" ]]; then
+    base_model_path="${local_sft_base}"
+    default_tokenizer_model_path="${base_model_path}"
+    echo "[duet][LoKU] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    echo "[duet][LoKU] Using Hugging Face base checkpoint ${base_model_path}"
+fi
+
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
+
+extra_train_args=()
+extra_importance_args=()
+extra_eval_tokenizer_args=()
+if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
+    extra_importance_args+=(+model.model_args.subfolder=${sft_subfolder})
+fi
+if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_importance_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_tokenizer_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/duet/loku_lora.yaml"
+trainer="LoKU"
+
+output_root="${repo_root}/saves/unlearn/duet/loku"
+importance_root="${repo_root}/saves/importances/duet/loku"
+mkdir -p "${output_root}" "${importance_root}"
+
+# Match current DUET default behavior used by NPO-SAM/FALCON scripts.
+export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
+set_forget_retain_splits
+
+per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
+gradient_accumulation_steps=${GRAD_ACCUM:-32}
+importance_batch_size=${IMPORTANCE_BATCH_SIZE:-1}
+importance_max_steps=${IMPORTANCE_MAX_STEPS:-0}
+eval_batch_size=${EVAL_BATCH_SIZE:-8}
+num_train_epochs=${NUM_EPOCHS:-5}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-4}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+raw_ihl_alphas="${IHL_ALPHAS:-1.0}"
+raw_ihl_alphas="${raw_ihl_alphas//,/ }"
+raw_ihl_alphas="${raw_ihl_alphas//\"/}"
+raw_ihl_alphas="${raw_ihl_alphas//\'/}"
+read -r -a ihl_alphas <<< "${raw_ihl_alphas}"
+
+raw_alphas="${ALPHAS:-1.0}"
+raw_alphas="${raw_alphas//,/ }"
+raw_alphas="${raw_alphas//\"/}"
+raw_alphas="${raw_alphas//\'/}"
+read -r -a alphas <<< "${raw_alphas}"
+
+raw_gammas="${GAMMAS:-1.0}"
+raw_gammas="${raw_gammas//,/ }"
+raw_gammas="${raw_gammas//\"/}"
+raw_gammas="${raw_gammas//\'/}"
+read -r -a gammas <<< "${raw_gammas}"
+
+fila_eps="${FILA_EPS:-1e-5}"
+fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
+fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
+run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"
+
+targets_tag="${LOKU_TARGETS_TAG:-all_lora_targets}"
+force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
+force_rerun="${FORCE_RERUN:-0}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
+
+export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
+
+for split in "${forget_retain_splits[@]}"; do
+    read -r forget_split retain_split forget_label <<< "${split}"
+    if [[ -z "${forget_label:-}" ]]; then
+        forget_label="${forget_split}"
+    fi
+
+    imp_path="${importance_root}/${base_model}_${forget_label}_${retain_split}_${targets_tag}.pt"
+    if [[ ! -f "${imp_path}" || "${force_importance}" == "1" ]]; then
+        echo "[duet][LoKU] Measuring importance -> ${imp_path}"
+        python src/tools/loku_measure_importance.py \
+            --config-name unlearn.yaml \
+            --experiment=${experiment} \
+            --output-path="${imp_path}" \
+            --max-steps=${importance_max_steps} \
+            --batch-size=${importance_batch_size} \
+            --seed=${TRAIN_SEED:-42} \
+            -- \
+            model=${lora_model} \
+            forget_split=${forget_split} \
+            retain_split=${retain_split} \
+            model.model_args.pretrained_model_name_or_path=${base_model_path} \
+            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+            model.model_args.device_map=null \
+            ++model.model_args.low_cpu_mem_usage=true \
+            trainer.args.per_device_train_batch_size=${importance_batch_size} \
+            trainer.args.gradient_accumulation_steps=1 \
+            trainer.args.gradient_checkpointing=false \
+            trainer.args.num_train_epochs=1 \
+            retain_logs_path=null \
+            "${extra_importance_args[@]}"
+    fi
+
+    for lr in "${lrs[@]}"; do
+        for ihl_alpha in "${ihl_alphas[@]}"; do
+            ihl_tag=${ihl_alpha//./p}
+            for alpha in "${alphas[@]}"; do
+                alpha_tag=${alpha//./p}
+                for gamma in "${gammas[@]}"; do
+                    gamma_tag=${gamma//./p}
+                    for lora_r in "${lora_rs[@]}"; do
+                        for lora_alpha in "${lora_alphas[@]}"; do
+                            for lora_dropout in "${lora_dropouts[@]}"; do
+                                dropout_tag=${lora_dropout//./p}
+                                task_name=duet_${base_model}_${forget_label}_loku_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_ihla${ihl_tag}_alpha${alpha_tag}_gamma${gamma_tag}
+                                run_dir=${output_root}/${task_name}
+                                eval_dir=${run_dir}/evals
+                                summary_path=${eval_dir}/DUET_SUMMARY.json
+                                base_residual_dir=${run_dir}/${fila_base_subdir}
+
+                                if [[ -f "${summary_path}" && "${force_rerun}" != "1" ]]; then
+                                    echo "[duet][LoKU] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                    continue
+                                fi
+
+                                echo "[duet][LoKU] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                adapter_path=${run_dir}/adapter_model.safetensors
+                                if [[ ! -f "${adapter_path}" || ! -d "${base_residual_dir}" || "${force_rerun}" == "1" ]]; then
+                                    mkdir -p "${run_dir}"
+                                    python src/train.py --config-name=unlearn.yaml \
+                                        experiment=${experiment} \
+                                        trainer=${trainer} \
+                                        task_name=${task_name} \
+                                        model=${lora_model} \
+                                        forget_split=${forget_split} \
+                                        retain_split=${retain_split} \
+                                        model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                        model.model_args.device_map="auto" \
+                                        ++model.model_args.low_cpu_mem_usage=true \
+                                        model.lora_config.r=${lora_r} \
+                                        model.lora_config.lora_alpha=${lora_alpha} \
+                                        model.lora_config.lora_dropout=${lora_dropout} \
+                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                        trainer.args.num_train_epochs=${num_train_epochs} \
+                                        trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                        trainer.args.learning_rate=${lr} \
+                                        trainer.method_args.ihl_alpha=${ihl_alpha} \
+                                        trainer.method_args.alpha=${alpha} \
+                                        trainer.method_args.gamma=${gamma} \
+                                        trainer.method_args.retain_loss_type=NLL \
+                                        trainer.method_args.importance_file=${imp_path} \
+                                        trainer.method_args.fila_eps=${fila_eps} \
+                                        trainer.method_args.fila_adapter_name=${fila_adapter_name} \
+                                        trainer.method_args.fila_base_subdir=${fila_base_subdir} \
+                                        trainer.method_args.run_fila_sanity_check=${run_fila_sanity_check} \
+                                        retain_logs_path=null \
+                                        "${extra_train_args[@]}" \
+                                        paths.output_dir=${run_dir}
+                                fi
+
+                                mkdir -p "${eval_dir}"
+                                if [[ "${force_rerun}" == "1" ]]; then
+                                    rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
+                                fi
+
+                                eval_cmd=( \
+                                    experiment=eval/duet/default.yaml \
+                                    model=${lora_model} \
+                                    forget_split=${forget_split} \
+                                    holdout_split=${retain_split} \
+                                    task_name=${task_name} \
+                                    model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                    ++model.model_args.base_model_name_or_path=${base_residual_dir} \
+                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                    model.model_args.device_map="auto" \
+                                    ++model.model_args.low_cpu_mem_usage=true \
+                                    model.lora_config.r=${lora_r} \
+                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                    eval.duet.batch_size=${eval_batch_size} \
+                                    eval.duet.overwrite=true \
+                                    "${extra_eval_tokenizer_args[@]}" \
+                                    paths.output_dir=${eval_dir} \
+                                    retain_logs_path=null \
+                                )
+                                python src/eval.py "${eval_cmd[@]}"
+
+                                if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                    if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                        rm -f "${run_dir}"/*.safetensors
+                                        echo "[duet][LoKU] Removed safetensors from ${run_dir}"
+                                    fi
+                                fi
+                            done
+                        done
+                    done
+                done
+            done
+        done
+    done
+done
diff --git a/scripts/popqa/loku_popqa.sh b/scripts/popqa/loku_popqa.sh
new file mode 100755
index 0000000..38d6ea2
--- /dev/null
+++ b/scripts/popqa/loku_popqa.sh
@@ -0,0 +1,251 @@
+#!/bin/bash
+
+set -euo pipefail
+
+script_dir=$(dirname "$(realpath "$0")")
+repo_root=$(realpath "${script_dir}/../..")
+
+export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
+echo "Master Port: $MASTER_PORT"
+
+base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
+lora_model="${MODEL_CONFIG:-${base_model}-lora}"
+hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
+local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/popqa/llama3.1-8b_full_5ep_ft_popqa}"
+sft_subfolder="${SFT_SUBFOLDER:-}"
+
+use_sft_base=${USE_SFT_BASE:-1}
+chat_template_tokenizer_path="${CHAT_TEMPLATE_TOKENIZER_PATH:-${repo_root}/assets/tokenizers/llama-3.1-8b-instruct-chat-template}"
+if [[ "${use_sft_base}" == "1" ]]; then
+    base_model_path="${local_sft_base}"
+    default_tokenizer_model_path="${base_model_path}"
+    default_tokenizer_subfolder="${sft_subfolder}"
+    if [[ "${sft_subfolder}" == "llama-3.1-8b-instruct-popqa-ft" ]]; then
+        default_tokenizer_model_path="${chat_template_tokenizer_path}"
+        default_tokenizer_subfolder=""
+    fi
+    echo "[popqa][LoKU] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    default_tokenizer_subfolder=""
+    echo "[popqa][LoKU] Using Hugging Face base checkpoint ${base_model_path}"
+fi
+
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${default_tokenizer_subfolder}}"
+
+extra_train_args=()
+extra_importance_args=()
+extra_eval_tokenizer_args=()
+if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
+    extra_importance_args+=(+model.model_args.subfolder=${sft_subfolder})
+fi
+if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_importance_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_tokenizer_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/popqa/loku_lora.yaml"
+trainer="LoKU"
+
+output_root="${repo_root}/saves/unlearn/popqa/loku"
+importance_root="${repo_root}/saves/importances/popqa/loku"
+mkdir -p "${output_root}" "${importance_root}"
+
+base_forget_retain_splits=(
+    "rare_forget5_sum fast_retain_500"
+    "popular_forget5_sum fast_retain_500"
+)
+
+if [[ "${MERGE_POPULARITY_FORGET:-0}" == "1" ]]; then
+    forget_retain_splits=(
+        "rare_forget5_sum+popular_forget5_sum fast_retain_500 forget5_sum"
+    )
+else
+    forget_retain_splits=("${base_forget_retain_splits[@]}")
+fi
+
+per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
+gradient_accumulation_steps=${GRAD_ACCUM:-32}
+importance_batch_size=${IMPORTANCE_BATCH_SIZE:-1}
+importance_max_steps=${IMPORTANCE_MAX_STEPS:-0}
+eval_batch_size=${EVAL_BATCH_SIZE:-8}
+num_train_epochs=${NUM_EPOCHS:-5}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-4}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+raw_ihl_alphas="${IHL_ALPHAS:-1.0}"
+raw_ihl_alphas="${raw_ihl_alphas//,/ }"
+raw_ihl_alphas="${raw_ihl_alphas//\"/}"
+raw_ihl_alphas="${raw_ihl_alphas//\'/}"
+read -r -a ihl_alphas <<< "${raw_ihl_alphas}"
+
+raw_alphas="${ALPHAS:-1.0}"
+raw_alphas="${raw_alphas//,/ }"
+raw_alphas="${raw_alphas//\"/}"
+raw_alphas="${raw_alphas//\'/}"
+read -r -a alphas <<< "${raw_alphas}"
+
+raw_gammas="${GAMMAS:-1.0}"
+raw_gammas="${raw_gammas//,/ }"
+raw_gammas="${raw_gammas//\"/}"
+raw_gammas="${raw_gammas//\'/}"
+read -r -a gammas <<< "${raw_gammas}"
+
+fila_eps="${FILA_EPS:-1e-5}"
+fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
+fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
+run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"
+
+targets_tag="${LOKU_TARGETS_TAG:-all_lora_targets}"
+force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
+force_rerun="${FORCE_RERUN:-0}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
+
+export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
+
+for split in "${forget_retain_splits[@]}"; do
+    read -r forget_split retain_split forget_label <<< "${split}"
+    if [[ -z "${forget_label:-}" ]]; then
+        forget_label="${forget_split}"
+    fi
+
+    imp_path="${importance_root}/${base_model}_${forget_label}_${retain_split}_${targets_tag}.pt"
+    if [[ ! -f "${imp_path}" || "${force_importance}" == "1" ]]; then
+        echo "[popqa][LoKU] Measuring importance -> ${imp_path}"
+        python src/tools/loku_measure_importance.py \
+            --config-name unlearn.yaml \
+            --experiment=${experiment} \
+            --output-path="${imp_path}" \
+            --max-steps=${importance_max_steps} \
+            --batch-size=${importance_batch_size} \
+            --seed=${TRAIN_SEED:-42} \
+            -- \
+            model=${lora_model} \
+            forget_split=${forget_split} \
+            retain_split=${retain_split} \
+            model.model_args.pretrained_model_name_or_path=${base_model_path} \
+            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+            model.model_args.device_map=null \
+            ++model.model_args.low_cpu_mem_usage=true \
+            trainer.args.per_device_train_batch_size=${importance_batch_size} \
+            trainer.args.gradient_accumulation_steps=1 \
+            trainer.args.gradient_checkpointing=false \
+            trainer.args.num_train_epochs=1 \
+            retain_logs_path=null \
+            "${extra_importance_args[@]}"
+    fi
+
+    for lr in "${lrs[@]}"; do
+        for ihl_alpha in "${ihl_alphas[@]}"; do
+            ihl_tag=${ihl_alpha//./p}
+            for alpha in "${alphas[@]}"; do
+                alpha_tag=${alpha//./p}
+                for gamma in "${gammas[@]}"; do
+                    gamma_tag=${gamma//./p}
+                    for lora_r in "${lora_rs[@]}"; do
+                        for lora_alpha in "${lora_alphas[@]}"; do
+                            for lora_dropout in "${lora_dropouts[@]}"; do
+                                dropout_tag=${lora_dropout//./p}
+                                task_name=popqa_${base_model}_${forget_label}_loku_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_ihla${ihl_tag}_alpha${alpha_tag}_gamma${gamma_tag}
+                                run_dir=${output_root}/${task_name}
+                                eval_dir=${run_dir}/evals
+                                summary_path=${eval_dir}/POPQA_SUMMARY.json
+                                base_residual_dir=${run_dir}/${fila_base_subdir}
+
+                                if [[ -f "${summary_path}" && "${force_rerun}" != "1" ]]; then
+                                    echo "[popqa][LoKU] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                    continue
+                                fi
+
+                                echo "[popqa][LoKU] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                adapter_path=${run_dir}/adapter_model.safetensors
+                                if [[ ! -f "${adapter_path}" || ! -d "${base_residual_dir}" || "${force_rerun}" == "1" ]]; then
+                                    mkdir -p "${run_dir}"
+                                    python src/train.py --config-name=unlearn.yaml \
+                                        experiment=${experiment} \
+                                        trainer=${trainer} \
+                                        task_name=${task_name} \
+                                        model=${lora_model} \
+                                        forget_split=${forget_split} \
+                                        retain_split=${retain_split} \
+                                        model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                        model.model_args.device_map="auto" \
+                                        ++model.model_args.low_cpu_mem_usage=true \
+                                        model.lora_config.r=${lora_r} \
+                                        model.lora_config.lora_alpha=${lora_alpha} \
+                                        model.lora_config.lora_dropout=${lora_dropout} \
+                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                        trainer.args.num_train_epochs=${num_train_epochs} \
+                                        trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                        trainer.args.learning_rate=${lr} \
+                                        trainer.method_args.ihl_alpha=${ihl_alpha} \
+                                        trainer.method_args.alpha=${alpha} \
+                                        trainer.method_args.gamma=${gamma} \
+                                        trainer.method_args.retain_loss_type=NLL \
+                                        trainer.method_args.importance_file=${imp_path} \
+                                        trainer.method_args.fila_eps=${fila_eps} \
+                                        trainer.method_args.fila_adapter_name=${fila_adapter_name} \
+                                        trainer.method_args.fila_base_subdir=${fila_base_subdir} \
+                                        trainer.method_args.run_fila_sanity_check=${run_fila_sanity_check} \
+                                        retain_logs_path=null \
+                                        "${extra_train_args[@]}" \
+                                        paths.output_dir=${run_dir}
+                                fi
+
+                                mkdir -p "${eval_dir}"
+                                if [[ "${force_rerun}" == "1" ]]; then
+                                    rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
+                                fi
+
+                                eval_cmd=( \
+                                    experiment=eval/popqa/default.yaml \
+                                    model=${lora_model} \
+                                    forget_split=${forget_split} \
+                                    holdout_split=${retain_split} \
+                                    task_name=${task_name} \
+                                    model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                    ++model.model_args.base_model_name_or_path=${base_residual_dir} \
+                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                    model.model_args.device_map="auto" \
+                                    ++model.model_args.low_cpu_mem_usage=true \
+                                    model.lora_config.r=${lora_r} \
+                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                    eval.duet.batch_size=${eval_batch_size} \
+                                    eval.duet.overwrite=true \
+                                    "${extra_eval_tokenizer_args[@]}" \
+                                    paths.output_dir=${eval_dir} \
+                                    retain_logs_path=null \
+                                )
+                                python src/eval.py "${eval_cmd[@]}"
+
+                                if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                    if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                        rm -f "${run_dir}"/*.safetensors
+                                        echo "[popqa][LoKU] Removed safetensors from ${run_dir}"
+                                    fi
+                                fi
+                            done
+                        done
+                    done
+                done
+            done
+        done
+    done
+done
```

## Recent updates (2026-03-02)

```diff
diff --git a/scripts/duet/loku_duet.sh b/scripts/duet/loku_duet.sh
index 7665bbf..9557ee6 100755
--- a/scripts/duet/loku_duet.sh
+++ b/scripts/duet/loku_duet.sh
@@ -90,7 +90,13 @@ fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
 fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
 run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"
 
-targets_tag="${LOKU_TARGETS_TAG:-all_lora_targets}"
+loku_target_modules="${LOKU_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
+loku_weight_decay="${LOKU_WEIGHT_DECAY:-0.01}"
+loku_lr_scheduler_type="${LOKU_LR_SCHEDULER_TYPE:-linear}"
+loku_warmup_epochs="${LOKU_WARMUP_EPOCHS:-1.0}"
+loku_warmup_ratio="${LOKU_WARMUP_RATIO:-0.0}"
+
+targets_tag="${LOKU_TARGETS_TAG:-no_lm_head_lora_targets}"
 force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
 force_rerun="${FORCE_RERUN:-0}"
 
@@ -125,6 +131,7 @@ for split in "${forget_retain_splits[@]}"; do
             model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
             model.model_args.device_map=null \
             ++model.model_args.low_cpu_mem_usage=true \
+            "model.lora_config.target_modules=${loku_target_modules}" \
             trainer.args.per_device_train_batch_size=${importance_batch_size} \
             trainer.args.gradient_accumulation_steps=1 \
             trainer.args.gradient_checkpointing=false \
@@ -171,6 +178,7 @@ for split in "${forget_retain_splits[@]}"; do
                                         model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                         model.model_args.device_map="auto" \
                                         ++model.model_args.low_cpu_mem_usage=true \
+                                        "model.lora_config.target_modules=${loku_target_modules}" \
                                         model.lora_config.r=${lora_r} \
                                         model.lora_config.lora_alpha=${lora_alpha} \
                                         model.lora_config.lora_dropout=${lora_dropout} \
@@ -179,6 +187,10 @@ for split in "${forget_retain_splits[@]}"; do
                                         trainer.args.num_train_epochs=${num_train_epochs} \
                                         trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                         trainer.args.learning_rate=${lr} \
+                                        trainer.args.weight_decay=${loku_weight_decay} \
+                                        trainer.args.lr_scheduler_type=${loku_lr_scheduler_type} \
+                                        trainer.args.warmup_epochs=${loku_warmup_epochs} \
+                                        trainer.args.warmup_ratio=${loku_warmup_ratio} \
                                         trainer.method_args.ihl_alpha=${ihl_alpha} \
                                         trainer.method_args.alpha=${alpha} \
                                         trainer.method_args.gamma=${gamma} \
@@ -209,6 +221,7 @@ for split in "${forget_retain_splits[@]}"; do
                                     model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                     model.model_args.device_map="auto" \
                                     ++model.model_args.low_cpu_mem_usage=true \
+                                    "model.lora_config.target_modules=${loku_target_modules}" \
                                     model.lora_config.r=${lora_r} \
                                     model.lora_config.lora_alpha=${lora_alpha} \
                                     model.lora_config.lora_dropout=${lora_dropout} \
diff --git a/scripts/popqa/loku_popqa.sh b/scripts/popqa/loku_popqa.sh
index 38d6ea2..46998dc 100755
--- a/scripts/popqa/loku_popqa.sh
+++ b/scripts/popqa/loku_popqa.sh
@@ -105,7 +105,13 @@ fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
 fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
 run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"
 
-targets_tag="${LOKU_TARGETS_TAG:-all_lora_targets}"
+loku_target_modules="${LOKU_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
+loku_weight_decay="${LOKU_WEIGHT_DECAY:-0.01}"
+loku_lr_scheduler_type="${LOKU_LR_SCHEDULER_TYPE:-linear}"
+loku_warmup_epochs="${LOKU_WARMUP_EPOCHS:-1.0}"
+loku_warmup_ratio="${LOKU_WARMUP_RATIO:-0.0}"
+
+targets_tag="${LOKU_TARGETS_TAG:-no_lm_head_lora_targets}"
 force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
 force_rerun="${FORCE_RERUN:-0}"
 
@@ -140,6 +146,7 @@ for split in "${forget_retain_splits[@]}"; do
             model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
             model.model_args.device_map=null \
             ++model.model_args.low_cpu_mem_usage=true \
+            "model.lora_config.target_modules=${loku_target_modules}" \
             trainer.args.per_device_train_batch_size=${importance_batch_size} \
             trainer.args.gradient_accumulation_steps=1 \
             trainer.args.gradient_checkpointing=false \
@@ -186,6 +193,7 @@ for split in "${forget_retain_splits[@]}"; do
                                         model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                         model.model_args.device_map="auto" \
                                         ++model.model_args.low_cpu_mem_usage=true \
+                                        "model.lora_config.target_modules=${loku_target_modules}" \
                                         model.lora_config.r=${lora_r} \
                                         model.lora_config.lora_alpha=${lora_alpha} \
                                         model.lora_config.lora_dropout=${lora_dropout} \
@@ -194,6 +202,10 @@ for split in "${forget_retain_splits[@]}"; do
                                         trainer.args.num_train_epochs=${num_train_epochs} \
                                         trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                         trainer.args.learning_rate=${lr} \
+                                        trainer.args.weight_decay=${loku_weight_decay} \
+                                        trainer.args.lr_scheduler_type=${loku_lr_scheduler_type} \
+                                        trainer.args.warmup_epochs=${loku_warmup_epochs} \
+                                        trainer.args.warmup_ratio=${loku_warmup_ratio} \
                                         trainer.method_args.ihl_alpha=${ihl_alpha} \
                                         trainer.method_args.alpha=${alpha} \
                                         trainer.method_args.gamma=${gamma} \
@@ -224,6 +236,7 @@ for split in "${forget_retain_splits[@]}"; do
                                     model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                     model.model_args.device_map="auto" \
                                     ++model.model_args.low_cpu_mem_usage=true \
+                                    "model.lora_config.target_modules=${loku_target_modules}" \
                                     model.lora_config.r=${lora_r} \
                                     model.lora_config.lora_alpha=${lora_alpha} \
                                     model.lora_config.lora_dropout=${lora_dropout} \
diff --git a/src/model/fila.py b/src/model/fila.py
index 064839d..34a3e46 100644
--- a/src/model/fila.py
+++ b/src/model/fila.py
@@ -19,7 +19,7 @@ def canonicalize_weight_name(name: str) -> str:
 
 
 def _weight_matches_targets(weight_name: str, target_modules: Sequence[str]) -> bool:
-    return any(weight_name.endswith(f".{target}.weight") for target in target_modules)
+    return any(target in weight_name for target in target_modules)
 
 
 def get_lora_layer_map(
diff --git a/src/tools/loku_measure_importance.py b/src/tools/loku_measure_importance.py
index 4cb0dd7..aa8420b 100755
--- a/src/tools/loku_measure_importance.py
+++ b/src/tools/loku_measure_importance.py
@@ -81,6 +81,11 @@ def _resolve_target_modules(cfg) -> List[str]:
     target_modules = list(lora_cfg.get("target_modules", []) or [])
     if not target_modules:
         raise ValueError("model.lora_config.target_modules must be non-empty.")
+    if any(str(name).split(".")[-1] == "lm_head" for name in target_modules):
+        raise ValueError(
+            "LoKU importance measurement requires `lm_head` to be excluded from "
+            "model.lora_config.target_modules."
+        )
     return target_modules
 
 
@@ -95,6 +100,8 @@ def _prepare_model_and_data(cfg, args: argparse.Namespace):
     template_args = model_cfg.template_args
 
     model, tokenizer = get_model(model_cfg)
+    if hasattr(model, "config") and model.config is not None:
+        model.config.use_cache = False
     device = _select_device(args)
     if getattr(model, "hf_device_map", None) is None:
         model = model.to(device)
diff --git a/src/train.py b/src/train.py
index 30c9174..d1d2c73 100644
--- a/src/train.py
+++ b/src/train.py
@@ -19,6 +19,8 @@ def main(cfg: DictConfig):
     template_args = model_cfg.template_args
     assert model_cfg is not None, "Invalid model yaml passed in train config."
     model, tokenizer = get_model(model_cfg)
+    if hasattr(model, "config") and model.config is not None:
+        model.config.use_cache = False
 
     # Load Dataset
     data_cfg = cfg.data
diff --git a/src/trainer/unlearn/loku.py b/src/trainer/unlearn/loku.py
index 81a02d9..b7af0a2 100644
--- a/src/trainer/unlearn/loku.py
+++ b/src/trainer/unlearn/loku.py
@@ -70,6 +70,11 @@ class LoKU(GradDiff):
         target_modules = list(getattr(cfg, "target_modules", []) or [])
         if not target_modules:
             raise ValueError("LoKU FILA needs non-empty model.lora_config.target_modules.")
+        if any(str(name).split(".")[-1] == "lm_head" for name in target_modules):
+            raise ValueError(
+                "LoKU requires `lm_head` to be excluded from model.lora_config.target_modules "
+                "to match the official implementation."
+            )
 
         rank = int(getattr(cfg, "r", 0))
         if rank <= 0:
diff --git a/src/trainer/utils.py b/src/trainer/utils.py
index a74c45a..c3db135 100644
--- a/src/trainer/utils.py
+++ b/src/trainer/utils.py
@@ -150,29 +150,28 @@ def ihl_loss_from_logits(
     ignore_index: int = -100,
     alpha: float = 1.0,
 ) -> torch.Tensor:
-    """Inverted Hinge Loss (IHL) on next-token probabilities.
-
-    Minimizing this loss encourages the true-token probability to be lower than
-    the strongest alternative by a margin alpha.
-    """
+    """Inverted Hinge Loss aligned with the official LoKU helper flow."""
     shift_logits = logits[..., :-1, :].contiguous()
     shift_labels = labels[..., 1:].contiguous()
 
+    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
+    shift_labels = shift_labels.view(-1)
+
     valid_mask = shift_labels != ignore_index
     if not valid_mask.any():
         return shift_logits.new_zeros(())
 
-    probs = shift_logits.softmax(dim=-1)
-    probs = probs[valid_mask]  # [N, V]
-    targets = shift_labels[valid_mask]  # [N]
+    preds = shift_logits[valid_mask, :]
+    targets = shift_labels[valid_mask]
+
+    if not torch.all((preds >= 0) * (preds <= 1)):
+        preds = preds.softmax(dim=1)
 
-    p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
-    probs_other = probs.clone()
-    probs_other.scatter_(1, targets.unsqueeze(1), float("-inf"))
-    p_other = probs_other.max(dim=1).values  # [N]
-    margins = p_true - p_other
+    margins = preds.gather(1, targets.unsqueeze(1)).squeeze(1)
+    preds_other = preds.clone()
+    preds_other.scatter_(1, targets.unsqueeze(1), float("-inf"))
+    margins = margins - preds_other.max(dim=1).values
 
-    # Inverted hinge: alpha + margin (official LoKU convention).
     measures = (float(alpha) + margins).clamp_min(0.0)
     return measures.mean()
 
```

## Recent updates (2026-03-02, FILA memory path)

```diff
diff --git a/src/model/fila.py b/src/model/fila.py
index 34a3e46..cac0c6e 100644
--- a/src/model/fila.py
+++ b/src/model/fila.py
@@ -166,9 +166,8 @@ def apply_fila_initialization(
         device = base_weight.device
         weight_fp32 = base_weight.float()
         imp = (imp_f.float() / f_cnt) / (float(eps) + (imp_r.float() / r_cnt))
-        imp = imp.to(device=device, dtype=torch.float32)
-
         row_importance = imp.sum(dim=1).clamp_min(0.0).sqrt()
+        row_importance = row_importance.to(device=device, dtype=torch.float32)
         weighted_w = row_importance.unsqueeze(1) * weight_fp32
 
         a_weight = layer.lora_A[adapter_name].weight.data
```
