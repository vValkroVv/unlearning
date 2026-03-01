# R2D Integration Diff

Base commit: ac4e90a8019475d1f9837700ef980ee511237732 (before R2D integration)
Target: current working tree

```diff
diff --git a/src/trainer/__init__.py b/src/trainer/__init__.py
index 7f4ccdc..67edfff 100644
--- a/src/trainer/__init__.py
+++ b/src/trainer/__init__.py
@@ -20,6 +20,7 @@ from trainer.unlearn.ada_wgd import AdaWGD, AdaWGDCallback
 from trainer.unlearn.ada_pop import AdaPop
 from trainer.unlearn.pop_dynam_b_wga import PopDynamBWGA
 from trainer.unlearn.falcon import FALCON
+from trainer.unlearn.r2d import R2D
 
 
 import logging
@@ -109,3 +110,4 @@ _register_trainer(AdaWGD)
 _register_trainer(AdaPop)
 _register_trainer(PopDynamBWGA)
 _register_trainer(FALCON)
+_register_trainer(R2D)
diff --git a/configs/experiment/unlearn/duet/r2d_lora.yaml b/configs/experiment/unlearn/duet/r2d_lora.yaml
new file mode 100644
index 0000000..63c1808
--- /dev/null
+++ b/configs/experiment/unlearn/duet/r2d_lora.yaml
@@ -0,0 +1,65 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: R2D
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
+  anchor: retain
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
+  method_args:
+    noise_std: 0.0
+    noise_trainable_only: true
+
+task_name: duet_r2d_lora
diff --git a/configs/experiment/unlearn/popqa/r2d_lora.yaml b/configs/experiment/unlearn/popqa/r2d_lora.yaml
new file mode 100644
index 0000000..36e3ce7
--- /dev/null
+++ b/configs/experiment/unlearn/popqa/r2d_lora.yaml
@@ -0,0 +1,65 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: R2D
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
+  anchor: retain
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
+  method_args:
+    noise_std: 0.0
+    noise_trainable_only: true
+
+task_name: popqa_r2d_lora
diff --git a/configs/trainer/R2D.yaml b/configs/trainer/R2D.yaml
new file mode 100644
index 0000000..5be3b95
--- /dev/null
+++ b/configs/trainer/R2D.yaml
@@ -0,0 +1,13 @@
+defaults:
+  - finetune
+
+handler: R2D
+
+method_args:
+  noise_std: 0.0
+  noise_seed: null
+  noise_trainable_only: true
+
+  dp_epsilon: null
+  dp_delta: null
+  dp_sensitivity: null
diff --git a/scripts/duet/r2d_duet.sh b/scripts/duet/r2d_duet.sh
new file mode 100755
index 0000000..d3dcddb
--- /dev/null
+++ b/scripts/duet/r2d_duet.sh
@@ -0,0 +1,205 @@
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
+
+sft_subfolder="${SFT_SUBFOLDER:-}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
+
+use_sft_base=${USE_SFT_BASE:-1}
+if [[ "${use_sft_base}" == "1" ]]; then
+    trained_model_path="${local_sft_base}"
+    echo "[duet][R2D] Using finetuned model path ${trained_model_path}"
+else
+    trained_model_path="${hf_base_model_path}"
+    echo "[duet][R2D] Using HF base model path ${trained_model_path}"
+fi
+
+# Required rewind inputs:
+# - R2D_REWIND_CKPT_PATH
+# - or R2D_REWIND_SUBFOLDER
+# - or R2D_REWIND_STEP (resolved as checkpoint-${R2D_REWIND_STEP})
+rewind_model_path="${R2D_REWIND_CKPT_PATH:-${trained_model_path}}"
+rewind_subfolder="${R2D_REWIND_SUBFOLDER:-}"
+
+if [[ -z "${rewind_subfolder}" && -n "${R2D_REWIND_STEP:-}" ]]; then
+    if [[ -n "${sft_subfolder}" ]]; then
+        rewind_subfolder="${sft_subfolder}/checkpoint-${R2D_REWIND_STEP}"
+    else
+        rewind_subfolder="checkpoint-${R2D_REWIND_STEP}"
+    fi
+fi
+
+if [[ -z "${R2D_REWIND_CKPT_PATH:-}" && -z "${R2D_REWIND_SUBFOLDER:-}" && -z "${R2D_REWIND_STEP:-}" ]]; then
+    echo "[duet][R2D] ERROR: set R2D_REWIND_CKPT_PATH or R2D_REWIND_SUBFOLDER or R2D_REWIND_STEP."
+    exit 1
+fi
+
+echo "[duet][R2D] Rewind model path: ${rewind_model_path}"
+echo "[duet][R2D] Rewind subfolder : ${rewind_subfolder}"
+
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${trained_model_path}}"
+
+extra_train_args=()
+extra_eval_args=()
+if [[ -n "${rewind_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${rewind_subfolder})
+    extra_eval_args+=(+model.model_args.subfolder=${rewind_subfolder})
+fi
+if [[ -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/duet/r2d_lora.yaml"
+trainer="R2D"
+
+output_root="${repo_root}/saves/unlearn/duet/r2d"
+mkdir -p "${output_root}"
+
+set_forget_retain_splits
+
+per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
+gradient_accumulation_steps=${GRAD_ACCUM:-32}
+eval_batch_size=${EVAL_BATCH_SIZE:-8}
+num_train_epochs=${NUM_EPOCHS:-1}
+max_steps=${R2D_MAX_STEPS:-0}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-5}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+
+r2d_noise_std="${R2D_NOISE_STD:-0.0}"
+r2d_noise_seed="${R2D_NOISE_SEED:-42}"
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
+    for lr in "${lrs[@]}"; do
+        for lora_r in "${lora_rs[@]}"; do
+            for lora_alpha in "${lora_alphas[@]}"; do
+                for lora_dropout in "${lora_dropouts[@]}"; do
+                    dropout_tag=${lora_dropout//./p}
+                    rewind_tag="${R2D_REWIND_TAG:-rewind}"
+                    task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${r2d_noise_std}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                    run_dir=${output_root}/${task_name}
+                    eval_dir=${run_dir}/evals
+                    summary_path=${eval_dir}/DUET_SUMMARY.json
+
+                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                        echo "[duet][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
+                        continue
+                    fi
+
+                    echo "[duet][R2D] ${task_name}"
+                    echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
+                    echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps}"
+
+                    adapter_path=${run_dir}/adapter_model.safetensors
+                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                        mkdir -p "${run_dir}"
+
+                        train_cmd=( \
+                            --config-name=unlearn.yaml \
+                            experiment=${experiment} \
+                            trainer=${trainer} \
+                            task_name=${task_name} \
+                            model=${lora_model} \
+                            forget_split=${forget_split} \
+                            retain_split=${retain_split} \
+                            holdout_split=${retain_split} \
+                            model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
+                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                            model.model_args.device_map="auto" \
+                            model.model_args.low_cpu_mem_usage=true \
+                            model.lora_config.r=${lora_r} \
+                            model.lora_config.lora_alpha=${lora_alpha} \
+                            model.lora_config.lora_dropout=${lora_dropout} \
+                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                            trainer.args.num_train_epochs=${num_train_epochs} \
+                            trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                            trainer.args.learning_rate=${lr} \
+                            trainer.args.lr_scheduler_type=constant \
+                            trainer.args.warmup_ratio=0.0 \
+                            trainer.args.save_strategy=no \
+                            trainer.args.eval_strategy=no \
+                            trainer.args.do_eval=false \
+                            trainer.args.eval_on_start=false \
+                            trainer.args.report_to=none \
+                            trainer.method_args.noise_std=${r2d_noise_std} \
+                            trainer.method_args.noise_seed=${r2d_noise_seed} \
+                            trainer.method_args.noise_trainable_only=true \
+                            retain_logs_path=null \
+                            "${extra_train_args[@]}" \
+                            paths.output_dir=${run_dir} \
+                        )
+                        if [[ "${max_steps}" != "0" ]]; then
+                            train_cmd+=(trainer.args.max_steps=${max_steps})
+                        fi
+
+                        python src/train.py "${train_cmd[@]}"
+                    fi
+
+                    mkdir -p "${eval_dir}"
+                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                        rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
+                    fi
+
+                    eval_cmd=( \
+                        experiment=eval/duet/default.yaml \
+                        model=${lora_model} \
+                        forget_split=${forget_split} \
+                        holdout_split=${retain_split} \
+                        task_name=${task_name} \
+                        model.model_args.pretrained_model_name_or_path=${run_dir} \
+                        model.model_args.base_model_name_or_path=${rewind_model_path} \
+                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                        model.model_args.device_map="auto" \
+                        model.model_args.low_cpu_mem_usage=true \
+                        model.lora_config.r=${lora_r} \
+                        model.lora_config.lora_alpha=${lora_alpha} \
+                        model.lora_config.lora_dropout=${lora_dropout} \
+                        eval.duet.batch_size=${eval_batch_size} \
+                        eval.duet.overwrite=true \
+                        "${extra_eval_args[@]}" \
+                        paths.output_dir=${eval_dir} \
+                        retain_logs_path=null \
+                    )
+                    python src/eval.py "${eval_cmd[@]}"
+
+                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                            rm -f "${run_dir}"/*.safetensors
+                            echo "[duet][R2D] Removed safetensors from ${run_dir}"
+                        fi
+                    fi
+                done
+            done
+        done
+    done
+done
diff --git a/scripts/popqa/r2d_popqa.sh b/scripts/popqa/r2d_popqa.sh
new file mode 100755
index 0000000..0dd307c
--- /dev/null
+++ b/scripts/popqa/r2d_popqa.sh
@@ -0,0 +1,216 @@
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
+
+sft_subfolder="${SFT_SUBFOLDER:-}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"
+chat_template_tokenizer_path="${CHAT_TEMPLATE_TOKENIZER_PATH:-${repo_root}/assets/tokenizers/llama-3.1-8b-instruct-chat-template}"
+
+use_sft_base=${USE_SFT_BASE:-1}
+if [[ "${use_sft_base}" == "1" ]]; then
+    trained_model_path="${local_sft_base}"
+    echo "[popqa][R2D] Using finetuned model path ${trained_model_path}"
+else
+    trained_model_path="${hf_base_model_path}"
+    echo "[popqa][R2D] Using HF base model path ${trained_model_path}"
+fi
+
+rewind_model_path="${R2D_REWIND_CKPT_PATH:-${trained_model_path}}"
+rewind_subfolder="${R2D_REWIND_SUBFOLDER:-}"
+
+if [[ -z "${rewind_subfolder}" && -n "${R2D_REWIND_STEP:-}" ]]; then
+    if [[ -n "${sft_subfolder}" ]]; then
+        rewind_subfolder="${sft_subfolder}/checkpoint-${R2D_REWIND_STEP}"
+    else
+        rewind_subfolder="checkpoint-${R2D_REWIND_STEP}"
+    fi
+fi
+
+if [[ -z "${R2D_REWIND_CKPT_PATH:-}" && -z "${R2D_REWIND_SUBFOLDER:-}" && -z "${R2D_REWIND_STEP:-}" ]]; then
+    echo "[popqa][R2D] ERROR: set R2D_REWIND_CKPT_PATH or R2D_REWIND_SUBFOLDER or R2D_REWIND_STEP."
+    exit 1
+fi
+
+echo "[popqa][R2D] Rewind model path: ${rewind_model_path}"
+echo "[popqa][R2D] Rewind subfolder : ${rewind_subfolder}"
+
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${trained_model_path}}"
+if [[ "${use_sft_base}" == "1" && "${sft_subfolder}" == "llama-3.1-8b-instruct-popqa-ft" ]]; then
+    tokenizer_model_path="${chat_template_tokenizer_path}"
+    tokenizer_subfolder=""
+fi
+
+extra_train_args=()
+extra_eval_args=()
+if [[ -n "${rewind_subfolder}" ]]; then
+    extra_train_args+=(+model.model_args.subfolder=${rewind_subfolder})
+    extra_eval_args+=(+model.model_args.subfolder=${rewind_subfolder})
+fi
+if [[ "${use_sft_base}" == "1" && -n "${tokenizer_subfolder}" ]]; then
+    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
+fi
+
+experiment="unlearn/popqa/r2d_lora.yaml"
+trainer="R2D"
+
+output_root="${repo_root}/saves/unlearn/popqa/r2d"
+mkdir -p "${output_root}"
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
+eval_batch_size=${EVAL_BATCH_SIZE:-8}
+num_train_epochs=${NUM_EPOCHS:-1}
+max_steps=${R2D_MAX_STEPS:-0}
+gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
+
+raw_lrs="${LRS:-1e-5}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+
+r2d_noise_std="${R2D_NOISE_STD:-0.0}"
+r2d_noise_seed="${R2D_NOISE_SEED:-42}"
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
+    for lr in "${lrs[@]}"; do
+        for lora_r in "${lora_rs[@]}"; do
+            for lora_alpha in "${lora_alphas[@]}"; do
+                for lora_dropout in "${lora_dropouts[@]}"; do
+                    dropout_tag=${lora_dropout//./p}
+                    rewind_tag="${R2D_REWIND_TAG:-rewind}"
+                    task_name=popqa_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${r2d_noise_std}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                    run_dir=${output_root}/${task_name}
+                    eval_dir=${run_dir}/evals
+                    summary_path=${eval_dir}/POPQA_SUMMARY.json
+
+                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                        echo "[popqa][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
+                        continue
+                    fi
+
+                    echo "[popqa][R2D] ${task_name}"
+                    echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
+                    echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps}"
+
+                    adapter_path=${run_dir}/adapter_model.safetensors
+                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                        mkdir -p "${run_dir}"
+
+                        train_cmd=( \
+                            --config-name=unlearn.yaml \
+                            experiment=${experiment} \
+                            trainer=${trainer} \
+                            task_name=${task_name} \
+                            model=${lora_model} \
+                            forget_split=${forget_split} \
+                            retain_split=${retain_split} \
+                            holdout_split=${retain_split} \
+                            model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
+                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                            model.model_args.device_map="auto" \
+                            model.model_args.low_cpu_mem_usage=true \
+                            model.lora_config.r=${lora_r} \
+                            model.lora_config.lora_alpha=${lora_alpha} \
+                            model.lora_config.lora_dropout=${lora_dropout} \
+                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                            trainer.args.num_train_epochs=${num_train_epochs} \
+                            trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                            trainer.args.learning_rate=${lr} \
+                            trainer.args.lr_scheduler_type=constant \
+                            trainer.args.warmup_ratio=0.0 \
+                            trainer.args.save_strategy=no \
+                            trainer.args.eval_strategy=no \
+                            trainer.args.do_eval=false \
+                            trainer.args.eval_on_start=false \
+                            trainer.args.report_to=none \
+                            trainer.method_args.noise_std=${r2d_noise_std} \
+                            trainer.method_args.noise_seed=${r2d_noise_seed} \
+                            trainer.method_args.noise_trainable_only=true \
+                            retain_logs_path=null \
+                            "${extra_train_args[@]}" \
+                            paths.output_dir=${run_dir} \
+                        )
+                        if [[ "${max_steps}" != "0" ]]; then
+                            train_cmd+=(trainer.args.max_steps=${max_steps})
+                        fi
+
+                        python src/train.py "${train_cmd[@]}"
+                    fi
+
+                    mkdir -p "${eval_dir}"
+                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                        rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
+                    fi
+
+                    eval_cmd=( \
+                        experiment=eval/popqa/default.yaml \
+                        model=${lora_model} \
+                        forget_split=${forget_split} \
+                        holdout_split=${retain_split} \
+                        task_name=${task_name} \
+                        model.model_args.pretrained_model_name_or_path=${run_dir} \
+                        model.model_args.base_model_name_or_path=${rewind_model_path} \
+                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                        model.model_args.device_map="auto" \
+                        model.model_args.low_cpu_mem_usage=true \
+                        model.lora_config.r=${lora_r} \
+                        model.lora_config.lora_alpha=${lora_alpha} \
+                        model.lora_config.lora_dropout=${lora_dropout} \
+                        eval.duet.batch_size=${eval_batch_size} \
+                        eval.duet.overwrite=true \
+                        "${extra_eval_args[@]}" \
+                        paths.output_dir=${eval_dir} \
+                        retain_logs_path=null \
+                    )
+                    python src/eval.py "${eval_cmd[@]}"
+
+                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                            rm -f "${run_dir}"/*.safetensors
+                            echo "[popqa][R2D] Removed safetensors from ${run_dir}"
+                        fi
+                    fi
+                done
+            done
+        done
+    done
+done
diff --git a/src/trainer/unlearn/r2d.py b/src/trainer/unlearn/r2d.py
new file mode 100644
index 0000000..d42ad70
--- /dev/null
+++ b/src/trainer/unlearn/r2d.py
@@ -0,0 +1,155 @@
+import logging
+import math
+from typing import Optional
+
+import torch
+
+from trainer.unlearn.base import UnlearnTrainer
+from trainer.utils import _filter_model_inputs
+
+logger = logging.getLogger(__name__)
+
+
+def _default_sigma_from_dp(epsilon: float, delta: float, sensitivity: float) -> float:
+    """
+    Dependency-free fallback for Gaussian mechanism calibration:
+        sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
+    """
+    if epsilon <= 0:
+        raise ValueError("epsilon must be > 0")
+    if not (0 < delta < 1):
+        raise ValueError("delta must be in (0, 1)")
+    if sensitivity < 0:
+        raise ValueError("sensitivity must be >= 0")
+    return float(sensitivity) * math.sqrt(2.0 * math.log(1.25 / float(delta))) / float(
+        epsilon
+    )
+
+
+@torch.no_grad()
+def _apply_noise(
+    model: torch.nn.Module,
+    sigma: float,
+    trainable_only: bool = True,
+) -> None:
+    for param in model.parameters():
+        if trainable_only and not param.requires_grad:
+            continue
+        if not torch.is_floating_point(param):
+            continue
+        noise = torch.randn_like(param, dtype=torch.float32) * sigma
+        param.add_(noise.to(dtype=param.dtype))
+
+
+@torch.no_grad()
+def add_gaussian_noise_to_weights(
+    model: torch.nn.Module,
+    sigma: float,
+    seed: Optional[int] = None,
+    trainable_only: bool = True,
+) -> None:
+    """
+    Output perturbation: add N(0, sigma^2) noise to model parameters.
+    """
+    if sigma <= 0:
+        return
+
+    if seed is None:
+        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)
+        return
+
+    devices = []
+    if torch.cuda.is_available():
+        devices = list(range(torch.cuda.device_count()))
+
+    with torch.random.fork_rng(devices=devices):
+        torch.manual_seed(seed)
+        if torch.cuda.is_available():
+            torch.cuda.manual_seed_all(seed)
+        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)
+
+
+class R2D(UnlearnTrainer):
+    """
+    Rewind-to-Delete (R2D) unlearning:
+    - train on retain-only NLL
+    - apply optional Gaussian output perturbation on final save
+    """
+
+    def __init__(
+        self,
+        *args,
+        noise_std: float = 0.0,
+        noise_seed: Optional[int] = None,
+        noise_trainable_only: bool = True,
+        dp_epsilon: Optional[float] = None,
+        dp_delta: Optional[float] = None,
+        dp_sensitivity: Optional[float] = None,
+        **kwargs,
+    ):
+        super().__init__(*args, **kwargs)
+        self.noise_std = float(noise_std)
+        self.noise_seed = noise_seed
+        self.noise_trainable_only = bool(noise_trainable_only)
+        self.dp_epsilon = dp_epsilon
+        self.dp_delta = dp_delta
+        self.dp_sensitivity = dp_sensitivity
+        self._noise_applied = False
+
+    def _resolve_sigma(self) -> float:
+        if self.noise_std < 0:
+            raise ValueError("noise_std must be >= 0")
+        if self.noise_std > 0:
+            return float(self.noise_std)
+
+        if (
+            self.dp_epsilon is not None
+            and self.dp_delta is not None
+            and self.dp_sensitivity is not None
+        ):
+            return _default_sigma_from_dp(
+                epsilon=float(self.dp_epsilon),
+                delta=float(self.dp_delta),
+                sensitivity=float(self.dp_sensitivity),
+            )
+
+        return 0.0
+
+    def compute_loss(self, model, inputs, return_outputs=False):
+        retain_inputs = _filter_model_inputs(inputs["retain"])
+        outputs = model(**retain_inputs)
+        loss = outputs.loss
+        return (loss, outputs) if return_outputs else loss
+
+    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
+        out_dir = output_dir or self.args.output_dir
+        sigma = self._resolve_sigma()
+
+        if (not self._noise_applied) and sigma > 0 and "checkpoint-" not in str(out_dir):
+            seed = self.noise_seed
+            if seed is None:
+                seed = int(getattr(self.args, "seed", 42))
+            process_index = int(getattr(self.args, "process_index", 0))
+            effective_seed = int(seed) + process_index
+            model_to_noise = self.model
+            if getattr(self, "accelerator", None) is not None:
+                try:
+                    model_to_noise = self.accelerator.unwrap_model(self.model)
+                except Exception:
+                    model_to_noise = self.model
+
+            logger.info(
+                "[R2D] Applying Gaussian output perturbation: sigma=%s trainable_only=%s seed=%s",
+                sigma,
+                self.noise_trainable_only,
+                effective_seed,
+            )
+            add_gaussian_noise_to_weights(
+                model=model_to_noise,
+                sigma=sigma,
+                seed=effective_seed,
+                trainable_only=self.noise_trainable_only,
+            )
+            self._noise_applied = True
+
+        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
```
