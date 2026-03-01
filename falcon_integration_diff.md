# FALCON Integration Diff

## Update: Official-mode alignment (commit `e30b533`, 2026-03-01)

This section captures the latest FALCON changes that align trainer behavior with the
official repository style while keeping backward-compatible paper mode.

### Changed files

- `src/trainer/unlearn/falcon.py`
- `configs/trainer/FALCON.yaml`

### `configs/trainer/FALCON.yaml` (updated defaults)

```yaml
method_args:
  # Backward-compatible GradDiff weights (used in paper mode).
  gamma: 1.0
  alpha: 1.0

  # Official FALCON defaults.
  temperature: 0.7
  retain_weight: 100.0
  steering_coeff: 20.0
  conflict_w: [0.8, 1.2]
  align_w: [0.1, 1.9]
  official_mode: true

  # Legacy paper-style knobs kept for compatibility (used when official_mode=false).
  k_svd: 16
  pov_alpha: 1.0
  pov_noise_std: 0.0
  pov_transform: tanh
  target_layer: 7
  conflict_cos_threshold: 0.0
  retain_mode: cosine
```

### `src/trainer/unlearn/falcon.py` (added args in `__init__`)

```python
def __init__(
    self,
    gamma: float = 1.0,
    alpha: float = 1.0,
    temperature: float = 0.7,
    k_svd: int = 16,
    pov_alpha: float = 1.0,
    pov_noise_std: float = 0.0,
    pov_transform: str = "tanh",
    target_layer: int = 7,
    conflict_cos_threshold: float = 0.0,
    retain_mode: str = "cosine",
    steering_coeff: float = 20.0,
    conflict_w: Sequence[float] = (0.8, 1.2),
    align_w: Sequence[float] = (0.1, 1.9),
    retain_weight: float = 100.0,
    official_mode: bool = True,
    *args,
    **kwargs,
):
    super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="NLL", *args, **kwargs)

    self.temperature = float(temperature)
    self.k_svd = int(k_svd)
    self.pov_alpha = float(pov_alpha)
    self.pov_noise_std = float(pov_noise_std)
    self.pov_transform = str(pov_transform).lower()
    self.target_layer = int(target_layer)
    self.conflict_cos_threshold = float(conflict_cos_threshold)
    self.retain_mode = str(retain_mode).lower()
    self.steering_coeff = float(steering_coeff)
    self.conflict_w = self._to_weight_pair(conflict_w, name="conflict_w")
    self.align_w = self._to_weight_pair(align_w, name="align_w")
    self.retain_weight = float(retain_weight)
    self.official_mode = bool(official_mode)
```

### Added official helper methods

```python
def _generate_steering_vector_official(self, model, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError(
            f"[FALCON] Expected hidden_states with shape (B, L, D), got {tuple(hidden_states.shape)}"
        )

    _ = model  # kept for parity with official signature
    _, _, d_model = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    with torch.no_grad():
        base_vector = torch.ones(1, 1, d_model, dtype=dtype, device=device)
        base_vector = base_vector / (torch.norm(base_vector) + 1e-6)

        states_matrix = hidden_states.detach().reshape(-1, d_model).to(torch.float32)
        try:
            _u, s_vals, vh = torch.linalg.svd(states_matrix, full_matrices=False)
        except RuntimeError:
            # Fallback keeps behavior deterministic on occasional SVD backend failures.
            q = max(1, min(states_matrix.shape))
            _u, s_vals, vh = torch.pca_lowrank(states_matrix, q=q, center=False)
            vh = vh.transpose(0, 1).contiguous()

        if s_vals.numel() == 0 or vh.numel() == 0:
            return base_vector.detach()

        k = min(1000, vh.shape[0])
        key_directions = vh[:k].to(device=device, dtype=dtype)

        s0 = s_vals[0].clamp_min(1e-12)
        weights = torch.sigmoid((s_vals[:k] / s0).to(device=device, dtype=dtype)) * (-1000.0)

        projection = torch.eye(d_model, dtype=dtype, device=device)
        weighted_dirs = key_directions.transpose(0, 1) * weights.unsqueeze(0)
        projection = projection - torch.matmul(weighted_dirs, key_directions)

        noise = torch.randn_like(projection) * 0.01
        projection = torch.tanh(projection + noise)

        final_vector = base_vector
        for _ in range(6):
            final_vector = torch.matmul(final_vector, projection)
            final_vector = torch.tanh(final_vector)
            final_vector = final_vector / (torch.norm(final_vector) + 1e-6)

    return final_vector.detach()

def _contrastive_loss_official(
    self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
) -> torch.Tensor:
    if anchor.dim() != 3 or positive.dim() != 3 or negatives.dim() != 3:
        raise ValueError(
            "[FALCON] Official contrastive loss expects (B, L, D) tensors for anchor/positive/negatives."
        )

    device = anchor.device
    positive = positive.to(device=device, dtype=anchor.dtype)
    negatives = negatives.to(device=device, dtype=anchor.dtype)

    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    negatives = negatives.unsqueeze(2)  # (B, L, 1, D)

    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True)  # (B, L, 1)
    neg_sim = torch.einsum("bld,blnd->bln", anchor, negatives)  # (B, L, 1)

    logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature  # (B, L, 2)
    labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=device)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))

def _resolve_grad_conflict_official(
    self,
    g_u: Sequence[Optional[torch.Tensor]],
    g_r: Sequence[Optional[torch.Tensor]],
) -> Tuple[list, list]:
    combined = []
    cos_sims = []

    for u_grad, r_grad in zip(g_u, g_r):
        if u_grad is None and r_grad is None:
            combined.append(None)
            cos_sims.append(0.0)
            continue

        if u_grad is None:
            u_grad = torch.zeros_like(r_grad)
        if r_grad is None:
            r_grad = torch.zeros_like(u_grad)

        cos = F.cosine_similarity(u_grad.view(-1), r_grad.view(-1), dim=0)
        cos_value = float(cos.detach().item())
        cos_sims.append(cos_value)

        if cos_value < 0.0:
            denom = torch.dot(r_grad.view(-1), r_grad.view(-1))
            if float(denom.abs().detach().item()) < 1e-12:
                proj_grad = u_grad
            else:
                coeff = torch.dot(u_grad.view(-1), r_grad.view(-1)) / denom
                proj_grad = u_grad - coeff * r_grad
            merged = self.conflict_w[0] * proj_grad + self.conflict_w[1] * r_grad
        else:
            merged = self.align_w[0] * u_grad + self.align_w[1] * r_grad

        combined.append(merged)

    return combined, cos_sims
```

### `training_step()` official-mode branch (added)

```python
if self.official_mode:
    steer_vec = self._generate_steering_vector_official(model, upd_f_acts.detach())
    steer_scaled = (
        steer_vec.expand_as(upd_f_acts).to(
            device=upd_f_acts.device, dtype=upd_f_acts.dtype
        )
        * self.steering_coeff
    )
    forget_loss = self._contrastive_loss_official(
        anchor=upd_f_acts,
        positive=steer_scaled,
        negatives=ref_f_acts.to(device=upd_f_acts.device, dtype=upd_f_acts.dtype),
    )
    ref_r_acts = ref_r_acts.to(device=upd_r_acts.device, dtype=upd_r_acts.dtype)
    retain_loss = 1.0 - F.cosine_similarity(upd_r_acts, ref_r_acts, dim=-1).mean()
    retain_loss = retain_loss * self.retain_weight

    merged_grads, cos_sims = self._resolve_grad_conflict_official(g_forget, g_retain)
    for param, grad in zip(trainable_params, merged_grads):
        if grad is None:
            continue
        grad = grad.detach() * grad_scale
        if param.grad is None:
            param.grad = grad
        else:
            param.grad.add_(grad)
else:
    # legacy paper-style path (kept)
    pass
```

### Full commit diff command

```bash
git show e30b533 -- src/trainer/unlearn/falcon.py configs/trainer/FALCON.yaml
```

Base commit: 9751ec152807e5bf47652e5506a37ebe460ce97d (before FALCON integration)
Target: current working tree (Falcon-related files only)

```diff
diff --git a/src/trainer/__init__.py b/src/trainer/__init__.py
index 1e4141e..aca89de 100644
--- a/src/trainer/__init__.py
+++ b/src/trainer/__init__.py
@@ -18,6 +18,7 @@ from trainer.unlearn.pdu import PDU
 from trainer.unlearn.ada_wgd import AdaWGD, AdaWGDCallback
 from trainer.unlearn.ada_pop import AdaPop
 from trainer.unlearn.pop_dynam_b_wga import PopDynamBWGA
+from trainer.unlearn.falcon import FALCON
 
 
 import logging
@@ -105,3 +106,4 @@ _register_trainer(PDU)
 _register_trainer(AdaWGD)
 _register_trainer(AdaPop)
 _register_trainer(PopDynamBWGA)
+_register_trainer(FALCON)
diff --git a/configs/eval/popqa.yaml b/configs/eval/popqa.yaml
index 8b351ff..0a48537 100644
--- a/configs/eval/popqa.yaml
+++ b/configs/eval/popqa.yaml
@@ -6,6 +6,7 @@ defaults:
     - holdout_qa_rouge
 
 handler: DUETEvaluator
+name: POPQA
 output_dir: ${paths.output_dir}
 metrics: {}
 overwrite: false
diff --git a/configs/experiment/unlearn/duet/falcon_lora.yaml b/configs/experiment/unlearn/duet/falcon_lora.yaml
new file mode 100644
index 0000000..b27d07d
--- /dev/null
+++ b/configs/experiment/unlearn/duet/falcon_lora.yaml
@@ -0,0 +1,62 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: FALCON
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
+    learning_rate: 1e-5
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
+
+task_name: duet_falcon_lora
diff --git a/configs/experiment/unlearn/popqa/falcon_lora.yaml b/configs/experiment/unlearn/popqa/falcon_lora.yaml
new file mode 100644
index 0000000..33d6f9e
--- /dev/null
+++ b/configs/experiment/unlearn/popqa/falcon_lora.yaml
@@ -0,0 +1,62 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: FALCON
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
+    learning_rate: 1e-5
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
+
+task_name: popqa_falcon_lora
diff --git a/configs/trainer/FALCON.yaml b/configs/trainer/FALCON.yaml
new file mode 100644
index 0000000..f34a10f
--- /dev/null
+++ b/configs/trainer/FALCON.yaml
@@ -0,0 +1,21 @@
+defaults:
+  - finetune
+
+handler: FALCON
+
+method_args:
+  gamma: 1.0
+  alpha: 1.0
+
+  temperature: 0.07
+
+  k_svd: 16
+  pov_alpha: 1.0
+  pov_noise_std: 0.0
+  pov_transform: tanh
+
+  target_layer: 7
+
+  conflict_cos_threshold: 0.0
+
+  retain_mode: cosine
diff --git a/scripts/duet/falcon_duet.sh b/scripts/duet/falcon_duet.sh
new file mode 100755
index 0000000..0cb7cd6
--- /dev/null
+++ b/scripts/duet/falcon_duet.sh
@@ -0,0 +1,338 @@
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
+    echo "[duet][FALCON] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    echo "[duet][FALCON] Using Hugging Face base checkpoint ${base_model_path}"
+fi
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
+extra_train_args=()
+extra_eval_args=()
+if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
+    extra_eval_args+=(+model.model_args.subfolder=${sft_subfolder})
+fi
+if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/duet/falcon_lora.yaml"
+trainer="FALCON"
+
+output_root="${repo_root}/saves/unlearn/duet/falcon"
+mkdir -p "${output_root}"
+
+# Match NPO/GA run style: merge rare+popular forget splits by default.
+export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
+set_forget_retain_splits
+
+per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
+gradient_accumulation_steps=${GRAD_ACCUM:-32}
+num_train_epochs=${NUM_EPOCHS:-5}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-5 5e-5 1e-4 5e-4 1e-3}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+raw_temps="${TEMPS:-0.07}"
+raw_temps="${raw_temps//,/ }"
+raw_temps="${raw_temps//\"/}"
+raw_temps="${raw_temps//\'/}"
+read -r -a temps <<< "${raw_temps}"
+
+raw_k_svds="${K_SVDS:-8,16}"
+raw_k_svds="${raw_k_svds//,/ }"
+raw_k_svds="${raw_k_svds//\"/}"
+raw_k_svds="${raw_k_svds//\'/}"
+read -r -a k_svds <<< "${raw_k_svds}"
+
+raw_pov_alphas="${POV_ALPHAS:-1.0}"
+raw_pov_alphas="${raw_pov_alphas//,/ }"
+raw_pov_alphas="${raw_pov_alphas//\"/}"
+raw_pov_alphas="${raw_pov_alphas//\'/}"
+read -r -a pov_alphas <<< "${raw_pov_alphas}"
+
+raw_pov_noise_stds="${POV_NOISE_STDS:-0.0}"
+raw_pov_noise_stds="${raw_pov_noise_stds//,/ }"
+raw_pov_noise_stds="${raw_pov_noise_stds//\"/}"
+raw_pov_noise_stds="${raw_pov_noise_stds//\'/}"
+read -r -a pov_noise_stds <<< "${raw_pov_noise_stds}"
+
+raw_pov_transforms="${POV_TRANSFORMS:-tanh}"
+raw_pov_transforms="${raw_pov_transforms//,/ }"
+raw_pov_transforms="${raw_pov_transforms//\"/}"
+raw_pov_transforms="${raw_pov_transforms//\'/}"
+read -r -a pov_transforms <<< "${raw_pov_transforms}"
+
+raw_target_layers="${TARGET_LAYERS:-7}"
+raw_target_layers="${raw_target_layers//,/ }"
+raw_target_layers="${raw_target_layers//\"/}"
+raw_target_layers="${raw_target_layers//\'/}"
+read -r -a target_layers <<< "${raw_target_layers}"
+
+raw_alphas="${ALPHAS:-2}"
+raw_alphas="${raw_alphas//,/ }"
+raw_alphas="${raw_alphas//\"/}"
+raw_alphas="${raw_alphas//\'/}"
+read -r -a alphas <<< "${raw_alphas}"
+
+raw_gammas="${GAMMAS:-2}"
+raw_gammas="${raw_gammas//,/ }"
+raw_gammas="${raw_gammas//\"/}"
+raw_gammas="${raw_gammas//\'/}"
+read -r -a gammas <<< "${raw_gammas}"
+
+raw_conflict_thresholds="${CONFLICT_COS_THRESHOLDS:-0.0}"
+raw_conflict_thresholds="${raw_conflict_thresholds//,/ }"
+raw_conflict_thresholds="${raw_conflict_thresholds//\"/}"
+raw_conflict_thresholds="${raw_conflict_thresholds//\'/}"
+read -r -a conflict_thresholds <<< "${raw_conflict_thresholds}"
+
+raw_retain_modes="${RETAIN_MODES:-cosine}"
+raw_retain_modes="${raw_retain_modes//,/ }"
+raw_retain_modes="${raw_retain_modes//\"/}"
+raw_retain_modes="${raw_retain_modes//\'/}"
+read -r -a retain_modes <<< "${raw_retain_modes}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
+
+# Optional MI-guided target-layer selection (paper-style preprocessing step).
+mi_select_layers=${MI_SELECT_LAYERS:-0}
+mi_model_cfg="${MI_MODEL_CFG:-${repo_root}/configs/model/Llama-3.1-8B-Instruct.yaml}"
+mi_model_path="${MI_MODEL_PATH:-${base_model_path}}"
+mi_model_subfolder="${MI_MODEL_SUBFOLDER-${sft_subfolder}}"
+mi_tokenizer_path="${MI_TOKENIZER_PATH:-${tokenizer_model_path}}"
+mi_tokenizer_subfolder="${MI_TOKENIZER_SUBFOLDER-${tokenizer_subfolder}}"
+mi_dataset_path="${MI_DATASET_PATH:-SwetieePawsss/DUET}"
+mi_question_key="${MI_QUESTION_KEY:-question}"
+mi_answer_key="${MI_ANSWER_KEY:-answer}"
+mi_answer_index="${MI_ANSWER_INDEX:-}"
+mi_max_length="${MI_MAX_LENGTH:-512}"
+mi_batch_size="${MI_BATCH_SIZE:-1}"
+mi_eta="${MI_ETA:-1.0}"
+mi_pca_var="${MI_PCA_VAR:-0.95}"
+mi_max_examples="${MI_MAX_EXAMPLES:-200}"
+raw_mi_topks="${MI_TOPK:-2}"
+raw_mi_topks="${raw_mi_topks//,/ }"
+raw_mi_topks="${raw_mi_topks//\"/}"
+raw_mi_topks="${raw_mi_topks//\'/}"
+read -r -a mi_topks <<< "${raw_mi_topks}"
+mi_seed="${MI_SEED:-0}"
+mi_device="${MI_DEVICE:-cuda}"
+mi_out_dir="${MI_OUT_DIR:-${output_root}/mi_layers}"
+
+export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
+
+for split in "${forget_retain_splits[@]}"; do
+    read -r forget_split retain_split forget_label <<< "${split}"
+    if [[ -z "${forget_label:-}" ]]; then
+        forget_label="${forget_split}"
+    fi
+    mi_loop_topks=("_none")
+    if [[ "${mi_select_layers}" == "1" ]]; then
+        mi_loop_topks=("${mi_topks[@]}")
+    fi
+
+    for mi_topk in "${mi_loop_topks[@]}"; do
+        split_target_layers=("${target_layers[@]}")
+        mi_topk_tag=""
+
+        if [[ "${mi_select_layers}" == "1" ]]; then
+            mi_forget_splits_raw="${MI_FORGET_SPLITS:-${forget_split//+/ }}"
+            mi_forget_splits_raw="${mi_forget_splits_raw//,/ }"
+            read -r -a mi_forget_splits <<< "${mi_forget_splits_raw}"
+            mi_retain_split="${MI_RETAIN_SPLIT:-${retain_split}}"
+            mkdir -p "${mi_out_dir}"
+            mi_out_json="${mi_out_dir}/${base_model}_${forget_label}_mi_topk${mi_topk}_layers.json"
+
+            mi_cmd=(
+                python "${repo_root}/src/tools/falcon_mi_select.py"
+                --model_cfg "${mi_model_cfg}"
+                --model_path "${mi_model_path}"
+                --tokenizer_path "${mi_tokenizer_path}"
+                --dataset_path "${mi_dataset_path}"
+                --forget_splits "${mi_forget_splits[@]}"
+                --retain_split "${mi_retain_split}"
+                --question_key "${mi_question_key}"
+                --answer_key "${mi_answer_key}"
+                --max_length "${mi_max_length}"
+                --batch_size "${mi_batch_size}"
+                --eta "${mi_eta}"
+                --pca_var "${mi_pca_var}"
+                --max_examples "${mi_max_examples}"
+                --topk "${mi_topk}"
+                --seed "${mi_seed}"
+                --device "${mi_device}"
+                --out_json "${mi_out_json}"
+                --print_layers
+                --quiet
+            )
+            if [[ -n "${mi_model_subfolder}" ]]; then
+                mi_cmd+=(--model_subfolder "${mi_model_subfolder}")
+            fi
+            if [[ -n "${mi_tokenizer_subfolder}" ]]; then
+                mi_cmd+=(--tokenizer_subfolder "${mi_tokenizer_subfolder}")
+            fi
+            if [[ -n "${mi_answer_index}" ]]; then
+                mi_cmd+=(--answer_index "${mi_answer_index}")
+            fi
+
+            echo "[duet][FALCON][MI] Selecting layers (topk=${mi_topk}) from splits: ${mi_forget_splits[*]} | retain: ${mi_retain_split}"
+            mi_layers_str="$("${mi_cmd[@]}")"
+            mi_layers_str="$(echo "${mi_layers_str}" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
+            if [[ -z "${mi_layers_str}" ]]; then
+                echo "[duet][FALCON][MI] Empty layer selection output; aborting."
+                exit 1
+            fi
+            read -r -a split_target_layers <<< "${mi_layers_str}"
+            mi_topk_tag="_mitk${mi_topk}"
+            echo "[duet][FALCON][MI] Selected TARGET_LAYERS=${mi_layers_str}"
+        fi
+
+        for lr in "${lrs[@]}"; do
+        for temp in "${temps[@]}"; do
+            temp_tag=${temp//./p}
+            for k_svd in "${k_svds[@]}"; do
+                for pov_alpha in "${pov_alphas[@]}"; do
+                    pov_alpha_tag=${pov_alpha//./p}
+                    for pov_noise_std in "${pov_noise_stds[@]}"; do
+                        pov_noise_tag=${pov_noise_std//./p}
+                        for pov_transform in "${pov_transforms[@]}"; do
+                            for target_layer in "${split_target_layers[@]}"; do
+                                for alpha in "${alphas[@]}"; do
+                                    alpha_tag=${alpha//./p}
+                                    for gamma in "${gammas[@]}"; do
+                                        gamma_tag=${gamma//./p}
+                                        for conflict_thr in "${conflict_thresholds[@]}"; do
+                                            conflict_tag=${conflict_thr//./p}
+                                            for retain_mode in "${retain_modes[@]}"; do
+                                                for lora_r in "${lora_rs[@]}"; do
+                                                        for lora_alpha in "${lora_alphas[@]}"; do
+                                                        for lora_dropout in "${lora_dropouts[@]}"; do
+                                                            dropout_tag=${lora_dropout//./p}
+
+                                                            task_name=duet_${base_model}_${forget_label}_falcon_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_t${temp_tag}_k${k_svd}_pova${pov_alpha_tag}_povn${pov_noise_tag}_pov${pov_transform}_layer${target_layer}${mi_topk_tag}_a${alpha_tag}_g${gamma_tag}_cth${conflict_tag}_rm${retain_mode}
+                                                            run_dir=${output_root}/${task_name}
+                                                            eval_dir=${run_dir}/evals
+                                                            summary_path=${eval_dir}/DUET_SUMMARY.json
+
+                                                            if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                                                echo "[duet][FALCON] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                                                continue
+                                                            fi
+
+                                                            echo "[duet][FALCON] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                                            adapter_path=${run_dir}/adapter_model.safetensors
+                                                            if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                                                mkdir -p "${run_dir}"
+                                                                python src/train.py --config-name=unlearn.yaml \
+                                                                    experiment=${experiment} \
+                                                                    trainer=${trainer} \
+                                                                    task_name=${task_name} \
+                                                                    model=${lora_model} \
+                                                                    forget_split=${forget_split} \
+                                                                    retain_split=${retain_split} \
+                                                                    model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                                    model.model_args.device_map="auto" \
+                                                                    model.model_args.low_cpu_mem_usage=true \
+                                                                    model.lora_config.r=${lora_r} \
+                                                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                                                    trainer.args.num_train_epochs=${num_train_epochs} \
+                                                                    trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                                                    trainer.args.learning_rate=${lr} \
+                                                                    trainer.method_args.temperature=${temp} \
+                                                                    trainer.method_args.k_svd=${k_svd} \
+                                                                    trainer.method_args.pov_alpha=${pov_alpha} \
+                                                                    trainer.method_args.pov_noise_std=${pov_noise_std} \
+                                                                    trainer.method_args.pov_transform=${pov_transform} \
+                                                                    trainer.method_args.target_layer=${target_layer} \
+                                                                    trainer.method_args.alpha=${alpha} \
+                                                                    trainer.method_args.gamma=${gamma} \
+                                                                    trainer.method_args.conflict_cos_threshold=${conflict_thr} \
+                                                                    trainer.method_args.retain_mode=${retain_mode} \
+                                                                    retain_logs_path=null \
+                                                                    "${extra_train_args[@]}" \
+                                                                    paths.output_dir=${run_dir}
+                                                            fi
+
+                                                            mkdir -p "${eval_dir}"
+                                                            if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                                                                rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
+                                                            fi
+
+                                                            eval_cmd=( \
+                                                                experiment=eval/duet/default.yaml \
+                                                                model=${lora_model} \
+                                                                forget_split=${forget_split} \
+                                                                holdout_split=${retain_split} \
+                                                                task_name=${task_name} \
+                                                                model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                                                model.model_args.base_model_name_or_path=${base_model_path} \
+                                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                                model.model_args.device_map="auto" \
+                                                                model.model_args.low_cpu_mem_usage=true \
+                                                                model.lora_config.r=${lora_r} \
+                                                                model.lora_config.lora_alpha=${lora_alpha} \
+                                                                model.lora_config.lora_dropout=${lora_dropout} \
+                                                                eval.duet.overwrite=true \
+                                                                "${extra_eval_args[@]}" \
+                                                                paths.output_dir=${eval_dir} \
+                                                                retain_logs_path=null \
+                                                            )
+                                                            python src/eval.py "${eval_cmd[@]}"
+
+                                                            if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                                                if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                                                    rm -f "${run_dir}"/*.safetensors
+                                                                    echo "[duet][FALCON] Removed safetensors from ${run_dir}"
+                                                                fi
+                                                            fi
+                done
+            done
+        done
+    done
+done
+                                        done
+                                    done
+                                done
+                            done
+                        done
+                    done
+                done
+            done
+        done
+    done
+done
diff --git a/scripts/popqa/falcon_popqa.sh b/scripts/popqa/falcon_popqa.sh
new file mode 100755
index 0000000..20a709c
--- /dev/null
+++ b/scripts/popqa/falcon_popqa.sh
@@ -0,0 +1,336 @@
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
+if [[ "${use_sft_base}" == "1" ]]; then
+    base_model_path="${local_sft_base}"
+    default_tokenizer_model_path="${base_model_path}"
+    echo "[popqa][FALCON] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    echo "[popqa][FALCON] Using Hugging Face base checkpoint ${base_model_path}"
+fi
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
+extra_train_args=()
+extra_eval_args=()
+if [[ "${use_sft_base}" == "1" && -n "${sft_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${sft_subfolder})
+    extra_eval_args+=(+model.model_args.subfolder=${sft_subfolder})
+fi
+if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/popqa/falcon_lora.yaml"
+trainer="FALCON"
+
+output_root="${repo_root}/saves/unlearn/popqa/falcon"
+mkdir -p "${output_root}"
+
+forget_retain_splits=(
+    "rare_forget5_sum fast_retain_500"
+    "popular_forget5_sum fast_retain_500"
+)
+
+per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
+gradient_accumulation_steps=${GRAD_ACCUM:-32}
+num_train_epochs=${NUM_EPOCHS:-5}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-5 5e-5 1e-4 5e-4 1e-3}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+raw_temps="${TEMPS:-0.07}"
+raw_temps="${raw_temps//,/ }"
+raw_temps="${raw_temps//\"/}"
+raw_temps="${raw_temps//\'/}"
+read -r -a temps <<< "${raw_temps}"
+
+raw_k_svds="${K_SVDS:-8,16}"
+raw_k_svds="${raw_k_svds//,/ }"
+raw_k_svds="${raw_k_svds//\"/}"
+raw_k_svds="${raw_k_svds//\'/}"
+read -r -a k_svds <<< "${raw_k_svds}"
+
+raw_pov_alphas="${POV_ALPHAS:-1.0}"
+raw_pov_alphas="${raw_pov_alphas//,/ }"
+raw_pov_alphas="${raw_pov_alphas//\"/}"
+raw_pov_alphas="${raw_pov_alphas//\'/}"
+read -r -a pov_alphas <<< "${raw_pov_alphas}"
+
+raw_pov_noise_stds="${POV_NOISE_STDS:-0.0}"
+raw_pov_noise_stds="${raw_pov_noise_stds//,/ }"
+raw_pov_noise_stds="${raw_pov_noise_stds//\"/}"
+raw_pov_noise_stds="${raw_pov_noise_stds//\'/}"
+read -r -a pov_noise_stds <<< "${raw_pov_noise_stds}"
+
+raw_pov_transforms="${POV_TRANSFORMS:-tanh}"
+raw_pov_transforms="${raw_pov_transforms//,/ }"
+raw_pov_transforms="${raw_pov_transforms//\"/}"
+raw_pov_transforms="${raw_pov_transforms//\'/}"
+read -r -a pov_transforms <<< "${raw_pov_transforms}"
+
+raw_target_layers="${TARGET_LAYERS:-7}"
+raw_target_layers="${raw_target_layers//,/ }"
+raw_target_layers="${raw_target_layers//\"/}"
+raw_target_layers="${raw_target_layers//\'/}"
+read -r -a target_layers <<< "${raw_target_layers}"
+
+raw_alphas="${ALPHAS:-2}"
+raw_alphas="${raw_alphas//,/ }"
+raw_alphas="${raw_alphas//\"/}"
+raw_alphas="${raw_alphas//\'/}"
+read -r -a alphas <<< "${raw_alphas}"
+
+raw_gammas="${GAMMAS:-2}"
+raw_gammas="${raw_gammas//,/ }"
+raw_gammas="${raw_gammas//\"/}"
+raw_gammas="${raw_gammas//\'/}"
+read -r -a gammas <<< "${raw_gammas}"
+
+raw_conflict_thresholds="${CONFLICT_COS_THRESHOLDS:-0.0}"
+raw_conflict_thresholds="${raw_conflict_thresholds//,/ }"
+raw_conflict_thresholds="${raw_conflict_thresholds//\"/}"
+raw_conflict_thresholds="${raw_conflict_thresholds//\'/}"
+read -r -a conflict_thresholds <<< "${raw_conflict_thresholds}"
+
+raw_retain_modes="${RETAIN_MODES:-cosine}"
+raw_retain_modes="${raw_retain_modes//,/ }"
+raw_retain_modes="${raw_retain_modes//\"/}"
+raw_retain_modes="${raw_retain_modes//\'/}"
+read -r -a retain_modes <<< "${raw_retain_modes}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
+
+# Optional MI-guided target-layer selection (paper-style preprocessing step).
+mi_select_layers=${MI_SELECT_LAYERS:-0}
+mi_model_cfg="${MI_MODEL_CFG:-${repo_root}/configs/model/Llama-3.1-8B-Instruct.yaml}"
+mi_model_path="${MI_MODEL_PATH:-${base_model_path}}"
+mi_model_subfolder="${MI_MODEL_SUBFOLDER-${sft_subfolder}}"
+mi_tokenizer_path="${MI_TOKENIZER_PATH:-${tokenizer_model_path}}"
+mi_tokenizer_subfolder="${MI_TOKENIZER_SUBFOLDER-${tokenizer_subfolder}}"
+mi_dataset_path="${MI_DATASET_PATH:-SwetieePawsss/exp_UNLamb}"
+mi_question_key="${MI_QUESTION_KEY:-question}"
+mi_answer_key="${MI_ANSWER_KEY:-possible_answers}"
+mi_answer_index="${MI_ANSWER_INDEX:-0}"
+mi_max_length="${MI_MAX_LENGTH:-512}"
+mi_batch_size="${MI_BATCH_SIZE:-1}"
+mi_eta="${MI_ETA:-1.0}"
+mi_pca_var="${MI_PCA_VAR:-0.95}"
+mi_max_examples="${MI_MAX_EXAMPLES:-200}"
+raw_mi_topks="${MI_TOPK:-2}"
+raw_mi_topks="${raw_mi_topks//,/ }"
+raw_mi_topks="${raw_mi_topks//\"/}"
+raw_mi_topks="${raw_mi_topks//\'/}"
+read -r -a mi_topks <<< "${raw_mi_topks}"
+mi_seed="${MI_SEED:-0}"
+mi_device="${MI_DEVICE:-cuda}"
+mi_out_dir="${MI_OUT_DIR:-${output_root}/mi_layers}"
+
+export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
+
+for split in "${forget_retain_splits[@]}"; do
+    forget_split=$(echo "$split" | cut -d' ' -f1)
+    retain_split=$(echo "$split" | cut -d' ' -f2)
+    mi_loop_topks=("_none")
+    if [[ "${mi_select_layers}" == "1" ]]; then
+        mi_loop_topks=("${mi_topks[@]}")
+    fi
+
+    for mi_topk in "${mi_loop_topks[@]}"; do
+        split_target_layers=("${target_layers[@]}")
+        mi_topk_tag=""
+
+        if [[ "${mi_select_layers}" == "1" ]]; then
+            mi_forget_splits_raw="${MI_FORGET_SPLITS:-${forget_split}}"
+            mi_forget_splits_raw="${mi_forget_splits_raw//,/ }"
+            read -r -a mi_forget_splits <<< "${mi_forget_splits_raw}"
+            mi_retain_split="${MI_RETAIN_SPLIT:-${retain_split}}"
+            mkdir -p "${mi_out_dir}"
+            mi_out_json="${mi_out_dir}/${base_model}_${forget_split}_mi_topk${mi_topk}_layers.json"
+
+            mi_cmd=(
+                python "${repo_root}/src/tools/falcon_mi_select.py"
+                --model_cfg "${mi_model_cfg}"
+                --model_path "${mi_model_path}"
+                --tokenizer_path "${mi_tokenizer_path}"
+                --dataset_path "${mi_dataset_path}"
+                --forget_splits "${mi_forget_splits[@]}"
+                --retain_split "${mi_retain_split}"
+                --question_key "${mi_question_key}"
+                --answer_key "${mi_answer_key}"
+                --max_length "${mi_max_length}"
+                --batch_size "${mi_batch_size}"
+                --eta "${mi_eta}"
+                --pca_var "${mi_pca_var}"
+                --max_examples "${mi_max_examples}"
+                --topk "${mi_topk}"
+                --seed "${mi_seed}"
+                --device "${mi_device}"
+                --out_json "${mi_out_json}"
+                --print_layers
+                --quiet
+            )
+            if [[ -n "${mi_model_subfolder}" ]]; then
+                mi_cmd+=(--model_subfolder "${mi_model_subfolder}")
+            fi
+            if [[ -n "${mi_tokenizer_subfolder}" ]]; then
+                mi_cmd+=(--tokenizer_subfolder "${mi_tokenizer_subfolder}")
+            fi
+            if [[ -n "${mi_answer_index}" ]]; then
+                mi_cmd+=(--answer_index "${mi_answer_index}")
+            fi
+
+            echo "[popqa][FALCON][MI] Selecting layers (topk=${mi_topk}) from splits: ${mi_forget_splits[*]} | retain: ${mi_retain_split}"
+            mi_layers_str="$("${mi_cmd[@]}")"
+            mi_layers_str="$(echo "${mi_layers_str}" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
+            if [[ -z "${mi_layers_str}" ]]; then
+                echo "[popqa][FALCON][MI] Empty layer selection output; aborting."
+                exit 1
+            fi
+            read -r -a split_target_layers <<< "${mi_layers_str}"
+            mi_topk_tag="_mitk${mi_topk}"
+            echo "[popqa][FALCON][MI] Selected TARGET_LAYERS=${mi_layers_str}"
+        fi
+
+        for lr in "${lrs[@]}"; do
+        for temp in "${temps[@]}"; do
+            temp_tag=${temp//./p}
+            for k_svd in "${k_svds[@]}"; do
+                for pov_alpha in "${pov_alphas[@]}"; do
+                    pov_alpha_tag=${pov_alpha//./p}
+                    for pov_noise_std in "${pov_noise_stds[@]}"; do
+                        pov_noise_tag=${pov_noise_std//./p}
+                        for pov_transform in "${pov_transforms[@]}"; do
+                            for target_layer in "${split_target_layers[@]}"; do
+                                for alpha in "${alphas[@]}"; do
+                                    alpha_tag=${alpha//./p}
+                                    for gamma in "${gammas[@]}"; do
+                                        gamma_tag=${gamma//./p}
+                                        for conflict_thr in "${conflict_thresholds[@]}"; do
+                                            conflict_tag=${conflict_thr//./p}
+                                            for retain_mode in "${retain_modes[@]}"; do
+                                                for lora_r in "${lora_rs[@]}"; do
+                                                        for lora_alpha in "${lora_alphas[@]}"; do
+                                                        for lora_dropout in "${lora_dropouts[@]}"; do
+                                                            dropout_tag=${lora_dropout//./p}
+
+                                                            task_name=popqa_${base_model}_${forget_split}_falcon_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_t${temp_tag}_k${k_svd}_pova${pov_alpha_tag}_povn${pov_noise_tag}_pov${pov_transform}_layer${target_layer}${mi_topk_tag}_a${alpha_tag}_g${gamma_tag}_cth${conflict_tag}_rm${retain_mode}
+                                                            run_dir=${output_root}/${task_name}
+                                                            eval_dir=${run_dir}/evals
+                                                            summary_path=${eval_dir}/POPQA_SUMMARY.json
+
+                                                            if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                                                echo "[popqa][FALCON] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                                                continue
+                                                            fi
+
+                                                            echo "[popqa][FALCON] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                                            adapter_path=${run_dir}/adapter_model.safetensors
+                                                            if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                                                mkdir -p "${run_dir}"
+                                                                python src/train.py --config-name=unlearn.yaml \
+                                                                    experiment=${experiment} \
+                                                                    trainer=${trainer} \
+                                                                    task_name=${task_name} \
+                                                                    model=${lora_model} \
+                                                                    forget_split=${forget_split} \
+                                                                    retain_split=${retain_split} \
+                                                                    model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                                    model.model_args.device_map="auto" \
+                                                                    model.model_args.low_cpu_mem_usage=true \
+                                                                    model.lora_config.r=${lora_r} \
+                                                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                                                    trainer.args.num_train_epochs=${num_train_epochs} \
+                                                                    trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                                                    trainer.args.learning_rate=${lr} \
+                                                                    trainer.method_args.temperature=${temp} \
+                                                                    trainer.method_args.k_svd=${k_svd} \
+                                                                    trainer.method_args.pov_alpha=${pov_alpha} \
+                                                                    trainer.method_args.pov_noise_std=${pov_noise_std} \
+                                                                    trainer.method_args.pov_transform=${pov_transform} \
+                                                                    trainer.method_args.target_layer=${target_layer} \
+                                                                    trainer.method_args.alpha=${alpha} \
+                                                                    trainer.method_args.gamma=${gamma} \
+                                                                    trainer.method_args.conflict_cos_threshold=${conflict_thr} \
+                                                                    trainer.method_args.retain_mode=${retain_mode} \
+                                                                    retain_logs_path=null \
+                                                                    "${extra_train_args[@]}" \
+                                                                    paths.output_dir=${run_dir}
+                                                            fi
+
+                                                            mkdir -p "${eval_dir}"
+                                                            if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                                                                rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
+                                                            fi
+
+                                                            eval_cmd=( \
+                                                                experiment=eval/popqa/default.yaml \
+                                                                model=${lora_model} \
+                                                                forget_split=${forget_split} \
+                                                                holdout_split=${retain_split} \
+                                                                task_name=${task_name} \
+                                                                model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                                                model.model_args.base_model_name_or_path=${base_model_path} \
+                                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                                model.model_args.device_map="auto" \
+                                                                model.model_args.low_cpu_mem_usage=true \
+                                                                model.lora_config.r=${lora_r} \
+                                                                model.lora_config.lora_alpha=${lora_alpha} \
+                                                                model.lora_config.lora_dropout=${lora_dropout} \
+                                                                eval.duet.overwrite=true \
+                                                                "${extra_eval_args[@]}" \
+                                                                paths.output_dir=${eval_dir} \
+                                                                retain_logs_path=null \
+                                                            )
+                                                            python src/eval.py "${eval_cmd[@]}"
+
+                                                            if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                                                if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                                                    rm -f "${run_dir}"/*.safetensors
+                                                                    echo "[popqa][FALCON] Removed safetensors from ${run_dir}"
+                                                                fi
+                                                            fi
+                done
+            done
+        done
+    done
+done
+                                        done
+                                    done
+                                done
+                            done
+                        done
+                    done
+                done
+            done
+        done
+    done
+done
diff --git a/src/evals/duet.py b/src/evals/duet.py
index 7f55aa6..2b82479 100644
--- a/src/evals/duet.py
+++ b/src/evals/duet.py
@@ -3,4 +3,5 @@ from evals.base import Evaluator
 
 class DUETEvaluator(Evaluator):
     def __init__(self, eval_cfg, **kwargs):
-        super().__init__("DUET", eval_cfg, **kwargs)
+        eval_name = eval_cfg.get("name", "DUET")
+        super().__init__(eval_name, eval_cfg, **kwargs)
diff --git a/src/tools/falcon_mi_select.py b/src/tools/falcon_mi_select.py
new file mode 100644
index 0000000..9fb1945
--- /dev/null
+++ b/src/tools/falcon_mi_select.py
@@ -0,0 +1,523 @@
+#!/usr/bin/env python3
+"""
+MI-based layer selection utility for FALCON.
+
+Implements the paper-style workflow:
+- Per-layer mutual information (MI) between forget and retain activations.
+- Multi-domain aggregate objective:
+    I(l) = sum_i I(F_i^l ; R^l) + eta * sum_{i<j} I(F_i^l ; F_j^l)
+- PCA dimensionality reduction (variance threshold) before KDE.
+- Gaussian KDE entropy estimate with Scott bandwidth.
+
+Usage examples:
+  DUET (multi-domain forget):
+    python src/tools/falcon_mi_select.py \
+      --model_cfg configs/model/Llama-3.1-8B-Instruct.yaml \
+      --model_path /path/to/base_or_sft_checkpoint \
+      --dataset_path SwetieePawsss/DUET \
+      --forget_splits city_forget_rare_5 city_forget_popular_5 \
+      --retain_split city_fast_retain_500 \
+      --print_layers
+
+  POPQA (single-domain, list answers):
+    python src/tools/falcon_mi_select.py \
+      --model_cfg configs/model/Llama-3.1-8B-Instruct.yaml \
+      --model_path /path/to/base_or_sft_checkpoint \
+      --dataset_path SwetieePawsss/exp_UNLamb \
+      --forget_splits rare_forget5_sum \
+      --retain_split fast_retain_500 \
+      --answer_key possible_answers --answer_index 0 \
+      --print_layers
+"""
+
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import sys
+import time
+from pathlib import Path
+from typing import Dict, List, Optional
+
+import numpy as np
+import torch
+from omegaconf import OmegaConf, open_dict
+from scipy.stats import gaussian_kde
+from sklearn.decomposition import PCA
+from torch.utils.data import DataLoader
+
+# Allow running as `python src/tools/falcon_mi_select.py` without PYTHONPATH tweaks.
+SRC_ROOT = Path(__file__).resolve().parent.parent
+if str(SRC_ROOT) not in sys.path:
+    sys.path.insert(0, str(SRC_ROOT))
+
+from data.collators import DataCollatorForSupervisedDataset
+from data.qa import QADataset, QAAnswerIndexDataset
+from data.utils import IGNORE_INDEX
+from model import get_model
+
+EPS = 1e-12
+
+
+def _log_progress(message: str, quiet: bool) -> None:
+    """Emit progress logs to stderr while keeping stdout clean for --print_layers."""
+    if quiet:
+        return
+    ts = time.strftime("%Y-%m-%d %H:%M:%S")
+    print(f"[MI][{ts}] {message}", file=sys.stderr, flush=True)
+
+
+def _fmt_secs(seconds: float) -> str:
+    if seconds < 60:
+        return f"{seconds:.1f}s"
+    minutes, sec = divmod(seconds, 60.0)
+    if minutes < 60:
+        return f"{int(minutes)}m {sec:.1f}s"
+    hours, minutes = divmod(minutes, 60.0)
+    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"
+
+
+def _pool_answer_tokens(
+    hidden_states: torch.Tensor,
+    labels: Optional[torch.Tensor],
+    attention_mask: Optional[torch.Tensor],
+) -> torch.Tensor:
+    """
+    Pool one vector per sample from layer hidden states [B, S, H].
+    Prefer answer-token span (labels != IGNORE_INDEX), fallback to attention mask.
+    """
+    pooled = []
+    batch_size = hidden_states.size(0)
+    for b in range(batch_size):
+        if labels is not None:
+            token_mask = labels[b] != IGNORE_INDEX
+        elif attention_mask is not None:
+            token_mask = attention_mask[b].bool()
+        else:
+            token_mask = None
+
+        if token_mask is not None and token_mask.any():
+            pooled.append(hidden_states[b, token_mask].mean(dim=0))
+        elif attention_mask is not None and attention_mask[b].any():
+            last_idx = int(attention_mask[b].sum().item()) - 1
+            pooled.append(hidden_states[b, max(last_idx, 0)])
+        else:
+            pooled.append(hidden_states[b, -1])
+    return torch.stack(pooled, dim=0)
+
+
+def _pca_reduce(features: np.ndarray, pca_var: float) -> np.ndarray:
+    """
+    Reduce feature dimensionality while preserving `pca_var` variance.
+    """
+    features = np.asarray(features, dtype=np.float64)
+    if features.ndim != 2:
+        raise ValueError(f"Expected 2D features, got shape={features.shape}")
+
+    n_samples, n_dims = features.shape
+    if n_samples < 5 or n_dims <= 1:
+        return features
+
+    max_components = min(n_dims, n_samples - 1)
+    if max_components < 1:
+        return features
+
+    if pca_var >= 1.0:
+        n_components = min(int(pca_var), max_components)
+    else:
+        n_components = float(pca_var)
+
+    reduced = PCA(n_components=n_components, svd_solver="full").fit_transform(features)
+
+    # gaussian_kde requires feature_dim < sample_count for stable covariance inversion.
+    if reduced.shape[1] >= reduced.shape[0]:
+        reduced = reduced[:, : max(1, reduced.shape[0] - 1)]
+    return reduced
+
+
+def _kde_entropy(features: np.ndarray, seed: int) -> float:
+    """
+    KDE plug-in entropy estimate:
+        H(X) ~= - E[log p_hat(X)]
+    with Gaussian KDE using Scott's bandwidth rule.
+    """
+    features = np.asarray(features, dtype=np.float64)
+    if features.ndim != 2 or features.shape[0] < 5:
+        return float("inf")
+
+    if features.shape[1] >= features.shape[0]:
+        features = features[:, : max(1, features.shape[0] - 1)]
+
+    rng = np.random.default_rng(seed)
+    for jitter in (0.0, 1e-6, 1e-5):
+        try:
+            values = features
+            if jitter > 0.0:
+                values = values + jitter * rng.standard_normal(values.shape)
+            kde = gaussian_kde(values.T, bw_method="scott")
+            probs = kde.evaluate(values.T)
+            return float(-np.mean(np.log(probs + EPS)))
+        except Exception:
+            continue
+    return float("inf")
+
+
+def _mutual_information(
+    x: np.ndarray,
+    y: np.ndarray,
+    pca_var: float,
+    seed: int,
+) -> float:
+    """
+    Estimate I(X;Y) via H(X) + H(Y) - H([X,Y]) after PCA.
+    """
+    if len(x) < 10 or len(y) < 10:
+        return float("inf")
+
+    rng = np.random.default_rng(seed)
+    n = min(len(x), len(y))
+    idx_x = rng.choice(len(x), size=n, replace=False)
+    idx_y = rng.choice(len(y), size=n, replace=False)
+    x_n = x[idx_x]
+    y_n = y[idx_y]
+
+    x_r = _pca_reduce(x_n, pca_var=pca_var)
+    y_r = _pca_reduce(y_n, pca_var=pca_var)
+    joint_r = _pca_reduce(np.concatenate([x_n, y_n], axis=1), pca_var=pca_var)
+
+    hx = _kde_entropy(x_r, seed=seed + 11)
+    hy = _kde_entropy(y_r, seed=seed + 17)
+    hxy = _kde_entropy(joint_r, seed=seed + 23)
+    if not np.isfinite(hx + hy + hxy):
+        return float("inf")
+    return float(hx + hy - hxy)
+
+
+def _make_dataset(
+    dataset_path: str,
+    split: str,
+    tokenizer,
+    template_args,
+    question_key: str,
+    answer_key: str,
+    answer_index: Optional[int],
+    max_length: int,
+):
+    base_kwargs = dict(
+        hf_args={"path": dataset_path, "split": split},
+        template_args=template_args,
+        tokenizer=tokenizer,
+        question_key=question_key,
+        answer_key=answer_key,
+        max_length=max_length,
+    )
+    if answer_index is None:
+        return QADataset(**base_kwargs)
+    return QAAnswerIndexDataset(answer_index=answer_index, **base_kwargs)
+
+
+def _resolve_num_layers(model) -> int:
+    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
+        value = getattr(model.config, attr, None)
+        if value is not None:
+            return int(value)
+    raise ValueError("Could not determine number of hidden layers from model config.")
+
+
+@torch.inference_mode()
+def _collect_layer_representations(
+    model,
+    dataloader: DataLoader,
+    num_layers: int,
+    device: str,
+    max_examples: int,
+) -> Dict[int, np.ndarray]:
+    """
+    Collect one pooled activation vector per example for each transformer layer.
+    Returns: layer_idx -> [N, H] numpy array
+    """
+    model.eval()
+    reps: Dict[int, List[torch.Tensor]] = {layer: [] for layer in range(num_layers)}
+    seen = 0
+
+    for batch in dataloader:
+        if seen >= max_examples:
+            break
+
+        model_inputs = {
+            key: value.to(device)
+            for key, value in batch.items()
+            if key in {"input_ids", "attention_mask", "labels"}
+        }
+        outputs = model(
+            **model_inputs,
+            output_hidden_states=True,
+            use_cache=False,
+            return_dict=True,
+        )
+
+        hidden_states = outputs.hidden_states
+        if hidden_states is None:
+            raise RuntimeError("Model did not return hidden_states.")
+        if len(hidden_states) < num_layers + 1:
+            raise RuntimeError(
+                f"Expected >= {num_layers + 1} hidden states, got {len(hidden_states)}."
+            )
+
+        labels = model_inputs.get("labels")
+        attention_mask = model_inputs.get("attention_mask")
+        batch_size = model_inputs["input_ids"].size(0)
+        take = min(batch_size, max_examples - seen)
+
+        for layer in range(num_layers):
+            pooled = _pool_answer_tokens(
+                hidden_states[layer + 1][:take],
+                labels[:take] if labels is not None else None,
+                attention_mask[:take] if attention_mask is not None else None,
+            )
+            reps[layer].append(pooled.detach().float().cpu())
+
+        seen += take
+
+    packed = {}
+    for layer in range(num_layers):
+        if reps[layer]:
+            packed[layer] = torch.cat(reps[layer], dim=0).numpy()
+        else:
+            packed[layer] = np.zeros((0, 1), dtype=np.float32)
+    return packed
+
+
+def _parse_args():
+    parser = argparse.ArgumentParser(description="Select FALCON target layers via MI.")
+    parser.add_argument("--model_cfg", required=True)
+    parser.add_argument("--model_path", required=True)
+    parser.add_argument("--model_subfolder", default=None)
+    parser.add_argument("--tokenizer_path", default=None)
+    parser.add_argument("--tokenizer_subfolder", default=None)
+
+    parser.add_argument("--dataset_path", required=True)
+    parser.add_argument("--forget_splits", nargs="+", required=True)
+    parser.add_argument("--retain_split", required=True)
+
+    parser.add_argument("--question_key", default="question")
+    parser.add_argument("--answer_key", default="answer")
+    parser.add_argument("--answer_index", type=int, default=None)
+    parser.add_argument("--max_length", type=int, default=512)
+    parser.add_argument("--batch_size", type=int, default=1)
+
+    parser.add_argument("--eta", type=float, default=1.0)
+    parser.add_argument("--pca_var", type=float, default=0.95)
+    parser.add_argument("--max_examples", type=int, default=200)
+    parser.add_argument("--seed", type=int, default=0)
+    parser.add_argument(
+        "--log_every_layers",
+        type=int,
+        default=1,
+        help="Emit MI progress log every N layers (disabled by --quiet).",
+    )
+
+    parser.add_argument("--topk", type=int, default=1)
+    parser.add_argument("--device", default="cuda")
+    parser.add_argument("--out_json", default=None)
+    parser.add_argument("--print_layers", action="store_true")
+    parser.add_argument("--quiet", action="store_true")
+    return parser.parse_args()
+
+
+def main():
+    run_start = time.perf_counter()
+    args = _parse_args()
+    np.random.seed(args.seed)
+    torch.manual_seed(args.seed)
+
+    if args.device.startswith("cuda") and not torch.cuda.is_available():
+        args.device = "cpu"
+
+    _log_progress(
+        "Starting MI selection "
+        f"(seed={args.seed}, device={args.device}, max_examples={args.max_examples})",
+        quiet=args.quiet,
+    )
+
+    stage_start = time.perf_counter()
+    cfg = OmegaConf.load(args.model_cfg)
+    with open_dict(cfg):
+        cfg.model_args.pretrained_model_name_or_path = args.model_path
+        if args.model_subfolder:
+            cfg.model_args.subfolder = args.model_subfolder
+        if args.tokenizer_path:
+            cfg.tokenizer_args.pretrained_model_name_or_path = args.tokenizer_path
+        if args.tokenizer_subfolder:
+            cfg.tokenizer_args.subfolder = args.tokenizer_subfolder
+
+    model, tokenizer = get_model(cfg)
+    model.to(args.device)
+    num_layers = _resolve_num_layers(model)
+    _log_progress(
+        f"Loaded model/tokenizer with {num_layers} transformer layers in "
+        f"{_fmt_secs(time.perf_counter() - stage_start)}.",
+        quiet=args.quiet,
+    )
+    template_args = cfg.template_args
+
+    collator = DataCollatorForSupervisedDataset(tokenizer)
+
+    stage_start = time.perf_counter()
+    retain_dataset = _make_dataset(
+        dataset_path=args.dataset_path,
+        split=args.retain_split,
+        tokenizer=tokenizer,
+        template_args=template_args,
+        question_key=args.question_key,
+        answer_key=args.answer_key,
+        answer_index=args.answer_index,
+        max_length=args.max_length,
+    )
+    retain_loader = DataLoader(
+        retain_dataset,
+        batch_size=max(1, int(args.batch_size)),
+        shuffle=True,
+        collate_fn=collator,
+    )
+    retain_reps = _collect_layer_representations(
+        model=model,
+        dataloader=retain_loader,
+        num_layers=num_layers,
+        device=args.device,
+        max_examples=args.max_examples,
+    )
+    retain_count = int(retain_reps[0].shape[0]) if 0 in retain_reps else 0
+    _log_progress(
+        f"Collected retain representations ({retain_count} examples) in "
+        f"{_fmt_secs(time.perf_counter() - stage_start)}.",
+        quiet=args.quiet,
+    )
+
+    forget_subdomain_reps = []
+    for split_idx, forget_split in enumerate(args.forget_splits):
+        stage_start = time.perf_counter()
+        _log_progress(
+            f"Collecting forget representations for split "
+            f"'{forget_split}' ({split_idx + 1}/{len(args.forget_splits)}).",
+            quiet=args.quiet,
+        )
+        forget_dataset = _make_dataset(
+            dataset_path=args.dataset_path,
+            split=forget_split,
+            tokenizer=tokenizer,
+            template_args=template_args,
+            question_key=args.question_key,
+            answer_key=args.answer_key,
+            answer_index=args.answer_index,
+            max_length=args.max_length,
+        )
+        forget_loader = DataLoader(
+            forget_dataset,
+            batch_size=max(1, int(args.batch_size)),
+            shuffle=True,
+            collate_fn=collator,
+        )
+        forget_subdomain_reps.append(
+            _collect_layer_representations(
+                model=model,
+                dataloader=forget_loader,
+                num_layers=num_layers,
+                device=args.device,
+                max_examples=args.max_examples,
+            )
+        )
+        split_count = int(forget_subdomain_reps[-1][0].shape[0]) if 0 in forget_subdomain_reps[-1] else 0
+        _log_progress(
+            f"Finished split '{forget_split}' ({split_count} examples) in "
+            f"{_fmt_secs(time.perf_counter() - stage_start)}.",
+            quiet=args.quiet,
+        )
+
+    mi_by_layer: Dict[int, float] = {}
+    n_sub = len(forget_subdomain_reps)
+    mi_start = time.perf_counter()
+    log_every = max(1, int(args.log_every_layers))
+    _log_progress(
+        f"Computing MI scores across {num_layers} layers.",
+        quiet=args.quiet,
+    )
+    for layer in range(num_layers):
+        layer_start = time.perf_counter()
+        main_term = 0.0
+        for i in range(n_sub):
+            main_term += _mutual_information(
+                forget_subdomain_reps[i][layer],
+                retain_reps[layer],
+                pca_var=args.pca_var,
+                seed=args.seed + 100 * i + layer,
+            )
+
+        inter_term = 0.0
+        if n_sub > 1:
+            for i in range(n_sub):
+                for j in range(i + 1, n_sub):
+                    inter_term += _mutual_information(
+                        forget_subdomain_reps[i][layer],
+                        forget_subdomain_reps[j][layer],
+                        pca_var=args.pca_var,
+                        seed=args.seed + 10000 + 97 * i + 193 * j + layer,
+                    )
+
+        mi_by_layer[layer] = float(main_term + args.eta * inter_term)
+        layer_idx = layer + 1
+        if layer_idx == 1 or layer_idx % log_every == 0 or layer_idx == num_layers:
+            elapsed = time.perf_counter() - mi_start
+            avg_per_layer = elapsed / layer_idx
+            eta = avg_per_layer * (num_layers - layer_idx)
+            _log_progress(
+                f"Layer {layer}/{num_layers - 1} score={mi_by_layer[layer]:.6f} "
+                f"step={_fmt_secs(time.perf_counter() - layer_start)} "
+                f"elapsed={_fmt_secs(elapsed)} "
+                f"eta={_fmt_secs(eta)}.",
+                quiet=args.quiet,
+            )
+
+    layer_ranking = sorted(
+        mi_by_layer.keys(),
+        key=lambda layer: (not np.isfinite(mi_by_layer[layer]), mi_by_layer[layer]),
+    )
+    topk = max(1, min(int(args.topk), len(layer_ranking)))
+    selected_layers = layer_ranking[:topk]
+
+    payload = {
+        "selected_layers": selected_layers,
+        "mi_by_layer": {str(layer): float(score) for layer, score in mi_by_layer.items()},
+        "args": vars(args),
+    }
+
+    if args.out_json:
+        out_dir = os.path.dirname(args.out_json)
+        if out_dir:
+            os.makedirs(out_dir, exist_ok=True)
+        with open(args.out_json, "w", encoding="utf-8") as fout:
+            json.dump(payload, fout, indent=2)
+
+    if args.print_layers:
+        _log_progress(
+            f"Completed MI selection in {_fmt_secs(time.perf_counter() - run_start)}. "
+            f"Selected layers: {selected_layers}",
+            quiet=args.quiet,
+        )
+        print(" ".join(str(layer) for layer in selected_layers))
+        return
+
+    if not args.quiet:
+        _log_progress(
+            f"Completed MI selection in {_fmt_secs(time.perf_counter() - run_start)}.",
+            quiet=args.quiet,
+        )
+        print(f"Selected layers (top-{topk}): {selected_layers}")
+        print("MI scores (lower is better):")
+        for layer in layer_ranking:
+            print(f"  layer {layer:>2}: {mi_by_layer[layer]:.6f}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/trainer/unlearn/falcon.py b/src/trainer/unlearn/falcon.py
new file mode 100644
index 0000000..19f8fa1
--- /dev/null
+++ b/src/trainer/unlearn/falcon.py
@@ -0,0 +1,363 @@
+import re
+from contextlib import contextmanager
+from typing import Optional
+
+import torch
+import torch.nn.functional as F
+
+from trainer.unlearn.grad_diff import GradDiff
+
+try:
+    import deepspeed  # type: ignore
+
+    _HAS_DEEPSPEED = True
+except Exception:
+    deepspeed = None
+    _HAS_DEEPSPEED = False
+
+
+class FALCON(GradDiff):
+    """
+    FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment.
+
+    Integration choices for this repository:
+    - Uses token-level samples (labels != -100) to make contrastive losses meaningful
+      with per_device_train_batch_size=1.
+    - Exposes target_layer directly (manual substitute for MI-guided layer selection).
+    - Uses cosine retain alignment (paper Eq. 11) by default.
+    - Applies orthogonal gradient projection when cosine(g_forget, g_retain) is below
+      conflict_cos_threshold.
+    """
+
+    def __init__(
+        self,
+        gamma: float = 1.0,
+        alpha: float = 1.0,
+        temperature: float = 0.07,
+        k_svd: int = 16,
+        pov_alpha: float = 1.0,
+        pov_noise_std: float = 0.0,
+        pov_transform: str = "tanh",
+        target_layer: int = 7,
+        conflict_cos_threshold: float = 0.0,
+        retain_mode: str = "cosine",
+        *args,
+        **kwargs,
+    ):
+        # Keep GradDiff plumbing for gamma/alpha and ref model preparation helpers.
+        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="NLL", *args, **kwargs)
+
+        self.temperature = float(temperature)
+        self.k_svd = int(k_svd)
+        self.pov_alpha = float(pov_alpha)
+        self.pov_noise_std = float(pov_noise_std)
+        self.pov_transform = str(pov_transform).lower()
+        self.target_layer = int(target_layer)
+        self.conflict_cos_threshold = float(conflict_cos_threshold)
+        self.retain_mode = str(retain_mode).lower()
+
+        model_unwrapped = self._unwrap(self.model)
+        self.uses_lora = hasattr(model_unwrapped, "disable_adapter")
+
+        self.ref_model = None
+        if not self.uses_lora:
+            self.ref_model = self._prepare_ref_model(self.model)
+
+        self.model_module = self._find_layer_module(self.model, self.target_layer)
+        ref_for_hook = self.ref_model if self.ref_model is not None else self.model
+        self.ref_module = self._find_layer_module(ref_for_hook, self.target_layer)
+
+    def _unwrap(self, model):
+        if _HAS_DEEPSPEED and isinstance(model, deepspeed.DeepSpeedEngine):
+            return model.module
+        if hasattr(model, "module"):
+            return model.module
+        return model
+
+    def _find_layer_module(self, model, layer_idx: int):
+        model_unwrapped = self._unwrap(model)
+        candidates = []
+        for name, module in model_unwrapped.named_modules():
+            parts = name.split(".")
+            if len(parts) < 2:
+                continue
+            if parts[-1] != str(layer_idx):
+                continue
+            if parts[-2] in {"layers", "h", "blocks"}:
+                candidates.append((name, module))
+
+        if not candidates:
+            fallback_patterns = (
+                rf".*model\.layers\.{layer_idx}$",
+                rf".*model\.model\.layers\.{layer_idx}$",
+                rf".*base_model\.model\.layers\.{layer_idx}$",
+                rf".*base_model\.model\.model\.layers\.{layer_idx}$",
+            )
+            for name, module in model_unwrapped.named_modules():
+                if any(re.match(pat, name) for pat in fallback_patterns):
+                    candidates.append((name, module))
+
+        if not candidates:
+            raise ValueError(
+                f"[FALCON] Could not find module for target_layer={layer_idx}. "
+                "Inspect model.named_modules() for your architecture."
+            )
+
+        candidates.sort(key=lambda x: len(x[0]))
+        return candidates[0][1]
+
+    def _as_model_inputs(self, inputs):
+        allowed = {
+            "input_ids",
+            "attention_mask",
+            "labels",
+            "position_ids",
+            "token_type_ids",
+            "inputs_embeds",
+        }
+        return {k: v for k, v in inputs.items() if k in allowed}
+
+    def _forward_with_cache(self, model, inputs, module, no_grad: bool):
+        cache = []
+
+        def hook(_module, _inp, out):
+            cache.append(out[0] if isinstance(out, tuple) else out)
+            return None
+
+        handle = module.register_forward_hook(hook)
+        with torch.set_grad_enabled(not no_grad):
+            outputs = model(**inputs)
+        handle.remove()
+        if not cache:
+            raise RuntimeError("[FALCON] Forward hook did not capture activations.")
+        return cache[0], outputs
+
+    @contextmanager
+    def _frozen_forward_context(self):
+        if self.ref_model is not None:
+            yield self.ref_model
+            return
+
+        model_unwrapped = self._unwrap(self.model)
+        disable_adapter = getattr(model_unwrapped, "disable_adapter", None)
+        if disable_adapter is None:
+            yield self.model
+            return
+
+        was_training = self.model.training
+        try:
+            self.model.eval()
+            with disable_adapter():
+                yield self.model
+        finally:
+            if was_training:
+                self.model.train()
+
+    def _token_samples(
+        self,
+        acts: torch.Tensor,
+        labels: Optional[torch.Tensor],
+        attention_mask: Optional[torch.Tensor],
+    ) -> torch.Tensor:
+        if labels is not None:
+            mask = labels != -100
+            if mask.any():
+                return acts[mask]
+        if attention_mask is not None:
+            mask = attention_mask.bool()
+            if mask.any():
+                return acts[mask]
+        return acts[:, -1, :]
+
+    def _compute_povs(self, ref_samples: torch.Tensor, num_samples: int) -> torch.Tensor:
+        X = ref_samples.detach()
+        X = X - X.mean(dim=0, keepdim=True)
+        n, d = X.shape
+        k = max(1, min(self.k_svd, n, d))
+
+        Xf = X.float()
+        try:
+            _u, _s, vh = torch.linalg.svd(Xf, full_matrices=False)
+            V = vh[:k].transpose(0, 1).contiguous()
+        except RuntimeError:
+            V = torch.pca_lowrank(Xf, q=k, center=False)[2]
+
+        V = V.to(device=ref_samples.device, dtype=ref_samples.dtype)
+        r = torch.randn(
+            (num_samples, d), device=ref_samples.device, dtype=ref_samples.dtype
+        )
+
+        proj = (r @ V) @ V.transpose(0, 1)
+        pov = r - self.pov_alpha * proj
+
+        if self.pov_transform == "tanh":
+            pov = torch.tanh(pov)
+        elif self.pov_transform == "none":
+            pass
+        else:
+            raise ValueError(f"[FALCON] Unsupported pov_transform={self.pov_transform}")
+
+        if self.pov_noise_std > 0:
+            pov = pov + self.pov_noise_std * torch.randn_like(pov)
+
+        return F.normalize(pov, dim=-1)
+
+    def _forget_infonce(
+        self, anchor: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
+    ) -> torch.Tensor:
+        a = F.normalize(anchor, dim=-1)
+        p = F.normalize(positives, dim=-1)
+        n = F.normalize(negatives, dim=-1)
+
+        pos_sim = (a * p).sum(dim=-1, keepdim=True)
+        neg_sim = a @ n.transpose(0, 1)
+        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature
+        labels = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
+        return F.cross_entropy(logits.float(), labels)
+
+    def _retain_alignment_loss(self, upd: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
+        mode = self.retain_mode
+        if mode in {"cosine", "cos"}:
+            # Paper Eq. (11): L_R = 1 - mean cosine similarity
+            a = F.normalize(upd, dim=-1)
+            p = F.normalize(ref, dim=-1)
+            return (1.0 - (a * p).sum(dim=-1)).mean()
+        if mode == "mse":
+            return F.mse_loss(upd, ref)
+        if mode != "contrastive":
+            raise ValueError(f"[FALCON] Unsupported retain_mode={mode}")
+
+        a = F.normalize(upd, dim=-1)
+        p = F.normalize(ref, dim=-1)
+        n = a.shape[0]
+        if n <= 1:
+            return (1.0 - (a * p).sum(dim=-1)).mean()
+        logits = (a @ p.transpose(0, 1)) / self.temperature
+        labels = torch.arange(n, device=a.device)
+        return F.cross_entropy(logits.float(), labels)
+
+    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
+        if self.is_deepspeed_enabled:
+            raise NotImplementedError(
+                "[FALCON] DeepSpeed path is not implemented for manual gradient projection."
+            )
+        if getattr(self.accelerator, "num_processes", 1) > 1:
+            raise NotImplementedError(
+                "[FALCON] Multi-process training is not implemented for manual gradient projection."
+            )
+
+        model.train()
+        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
+            self.optimizer.train()
+
+        inputs = self._prepare_inputs(inputs)
+        forget_inputs = self._as_model_inputs(inputs["forget"])
+        retain_inputs = self._as_model_inputs(inputs["retain"])
+
+        with self.compute_loss_context_manager():
+            upd_f_acts, _ = self._forward_with_cache(
+                model, forget_inputs, module=self.model_module, no_grad=False
+            )
+            with self._frozen_forward_context() as frozen_model:
+                ref_f_acts, _ = self._forward_with_cache(
+                    frozen_model, forget_inputs, module=self.ref_module, no_grad=True
+                )
+
+            upd_f = self._token_samples(
+                upd_f_acts, forget_inputs.get("labels"), forget_inputs.get("attention_mask")
+            )
+            ref_f = self._token_samples(
+                ref_f_acts, forget_inputs.get("labels"), forget_inputs.get("attention_mask")
+            )
+            povs = self._compute_povs(ref_f, num_samples=upd_f.shape[0])
+            forget_loss = self._forget_infonce(anchor=upd_f, positives=povs, negatives=ref_f)
+
+            upd_r_acts, _ = self._forward_with_cache(
+                model, retain_inputs, module=self.model_module, no_grad=False
+            )
+            with self._frozen_forward_context() as frozen_model:
+                ref_r_acts, _ = self._forward_with_cache(
+                    frozen_model, retain_inputs, module=self.ref_module, no_grad=True
+                )
+
+            upd_r = self._token_samples(
+                upd_r_acts, retain_inputs.get("labels"), retain_inputs.get("attention_mask")
+            )
+            ref_r = self._token_samples(
+                ref_r_acts, retain_inputs.get("labels"), retain_inputs.get("attention_mask")
+            )
+            retain_loss = self._retain_alignment_loss(upd=upd_r, ref=ref_r.to(upd_r.device))
+
+        trainable_params = [p for p in model.parameters() if p.requires_grad]
+        if not trainable_params:
+            raise RuntimeError("[FALCON] No trainable parameters found.")
+
+        g_forget = torch.autograd.grad(
+            forget_loss,
+            trainable_params,
+            retain_graph=False,
+            create_graph=False,
+            allow_unused=True,
+        )
+        g_retain = torch.autograd.grad(
+            retain_loss,
+            trainable_params,
+            retain_graph=False,
+            create_graph=False,
+            allow_unused=True,
+        )
+
+        dot = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
+        nf = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
+        nr = torch.zeros((), device=forget_loss.device, dtype=torch.float32)
+        for gf, gr in zip(g_forget, g_retain):
+            if gf is None or gr is None:
+                continue
+            gf32 = gf.float()
+            gr32 = gr.float()
+            dot = dot + (gf32 * gr32).sum()
+            nf = nf + (gf32 * gf32).sum()
+            nr = nr + (gr32 * gr32).sum()
+
+        eps = 1e-12
+        cos = dot / (torch.sqrt(nf + eps) * torch.sqrt(nr + eps) + eps)
+        conflict = bool(
+            cos.detach().item() < self.conflict_cos_threshold and nr.detach().item() > 0.0
+        )
+        proj_coeff = (dot / (nr + eps)) if conflict else None
+
+        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
+        grad_scale = 1.0 / grad_acc_steps
+
+        for param, gf, gr in zip(trainable_params, g_forget, g_retain):
+            if gf is None and gr is None:
+                continue
+            if gf is None:
+                gf = torch.zeros_like(gr)
+            if gr is None:
+                gr = torch.zeros_like(gf)
+
+            if conflict and proj_coeff is not None:
+                gf = gf - proj_coeff.to(gf.dtype) * gr
+
+            grad = (self.gamma * gf + self.alpha * gr).detach() * grad_scale
+            if param.grad is None:
+                param.grad = grad
+            else:
+                param.grad.add_(grad)
+
+        total_loss = self.gamma * forget_loss + self.alpha * retain_loss
+
+        try:
+            self.log(
+                {
+                    "falcon_forget_loss": float(forget_loss.detach().item()),
+                    "falcon_retain_loss": float(retain_loss.detach().item()),
+                    "falcon_grad_cos": float(cos.detach().item()),
+                    "falcon_conflict": 1.0 if conflict else 0.0,
+                }
+            )
+        except Exception:
+            pass
+
+        return total_loss.detach() * grad_scale
```
