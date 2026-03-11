# NPOSAM Integration Diff

Base commit: be4b363a5ab231c348254b90fbf2e348c13a523c (before NPOSAM integration)
Target: current working tree

```diff
diff --git a/configs/experiment/unlearn/duet/npo_sam_lora.yaml b/configs/experiment/unlearn/duet/npo_sam_lora.yaml
new file mode 100644
index 0000000..f78ef32
--- /dev/null
+++ b/configs/experiment/unlearn/duet/npo_sam_lora.yaml
@@ -0,0 +1,62 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: NPOSAM
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
+task_name: duet_npo_sam_lora
diff --git a/configs/experiment/unlearn/popqa/npo_sam_lora.yaml b/configs/experiment/unlearn/popqa/npo_sam_lora.yaml
new file mode 100644
index 0000000..d8586c4
--- /dev/null
+++ b/configs/experiment/unlearn/popqa/npo_sam_lora.yaml
@@ -0,0 +1,62 @@
+# @package _global_
+
+defaults:
+  - override /model: Llama-3.1-8B-Instruct-lora
+  - override /trainer: NPOSAM
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
+task_name: popqa_npo_sam_lora
diff --git a/configs/trainer/NPOSAM.yaml b/configs/trainer/NPOSAM.yaml
new file mode 100644
index 0000000..2ab29b1
--- /dev/null
+++ b/configs/trainer/NPOSAM.yaml
@@ -0,0 +1,14 @@
+defaults:
+  - NPO
+
+handler: NPOSAM
+
+method_args:
+  beta: 0.5
+  alpha: 1.0
+  gamma: 1.0
+  retain_loss_type: NLL
+
+  sam_rho: 0.01
+  sam_adaptive: false
+  sam_eps: 1e-12
diff --git a/scripts/duet/npo_sam_duet.sh b/scripts/duet/npo_sam_duet.sh
new file mode 100755
index 0000000..39dce90
--- /dev/null
+++ b/scripts/duet/npo_sam_duet.sh
@@ -0,0 +1,211 @@
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
+    echo "[duet][NPOSAM] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    echo "[duet][NPOSAM] Using Hugging Face base checkpoint ${base_model_path}"
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
+experiment="unlearn/duet/npo_sam_lora.yaml"
+trainer="NPOSAM"
+
+output_root="${repo_root}/saves/unlearn/duet/npo_sam"
+mkdir -p "${output_root}"
+
+# Match FALCON default for DUET: merged rare+popular forget split.
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
+raw_betas="${BETAS:-0.5}"
+raw_betas="${raw_betas//,/ }"
+raw_betas="${raw_betas//\"/}"
+raw_betas="${raw_betas//\'/}"
+read -r -a betas <<< "${raw_betas}"
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
+raw_sam_rhos="${SAM_RHOS:-0.01}"
+raw_sam_rhos="${raw_sam_rhos//,/ }"
+raw_sam_rhos="${raw_sam_rhos//\"/}"
+raw_sam_rhos="${raw_sam_rhos//\'/}"
+read -r -a sam_rhos <<< "${raw_sam_rhos}"
+
+raw_sam_adaptives="${SAM_ADAPTIVES:-false}"
+raw_sam_adaptives="${raw_sam_adaptives//,/ }"
+raw_sam_adaptives="${raw_sam_adaptives//\"/}"
+raw_sam_adaptives="${raw_sam_adaptives//\'/}"
+read -r -a sam_adaptives <<< "${raw_sam_adaptives}"
+
+sam_eps="${SAM_EPS:-1e-12}"
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
+    for lr in "${lrs[@]}"; do
+        for beta in "${betas[@]}"; do
+            beta_tag=${beta//./p}
+            for alpha in "${alphas[@]}"; do
+                alpha_tag=${alpha//./p}
+                for gamma in "${gammas[@]}"; do
+                    gamma_tag=${gamma//./p}
+                    for sam_rho in "${sam_rhos[@]}"; do
+                        rho_tag=${sam_rho//./p}
+                        for sam_adaptive in "${sam_adaptives[@]}"; do
+                            adapt_tag="${sam_adaptive}"
+                            adapt_tag="${adapt_tag//true/T}"
+                            adapt_tag="${adapt_tag//false/F}"
+                            for lora_r in "${lora_rs[@]}"; do
+                                for lora_alpha in "${lora_alphas[@]}"; do
+                                    for lora_dropout in "${lora_dropouts[@]}"; do
+                                        dropout_tag=${lora_dropout//./p}
+                                        task_name=duet_${base_model}_${forget_label}_npo_sam_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_rho${rho_tag}_ad${adapt_tag}
+                                        run_dir=${output_root}/${task_name}
+                                        eval_dir=${run_dir}/evals
+                                        summary_path=${eval_dir}/DUET_SUMMARY.json
+
+                                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                            echo "[duet][NPOSAM] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                            continue
+                                        fi
+
+                                        echo "[duet][NPOSAM] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                        adapter_path=${run_dir}/adapter_model.safetensors
+                                        if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                            mkdir -p "${run_dir}"
+                                            python src/train.py --config-name=unlearn.yaml \
+                                                experiment=${experiment} \
+                                                trainer=${trainer} \
+                                                task_name=${task_name} \
+                                                model=${lora_model} \
+                                                forget_split=${forget_split} \
+                                                retain_split=${retain_split} \
+                                                model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                model.model_args.device_map="auto" \
+                                                model.model_args.low_cpu_mem_usage=true \
+                                                model.lora_config.r=${lora_r} \
+                                                model.lora_config.lora_alpha=${lora_alpha} \
+                                                model.lora_config.lora_dropout=${lora_dropout} \
+                                                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                                trainer.args.num_train_epochs=${num_train_epochs} \
+                                                trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                                trainer.args.learning_rate=${lr} \
+                                                trainer.method_args.beta=${beta} \
+                                                trainer.method_args.alpha=${alpha} \
+                                                trainer.method_args.gamma=${gamma} \
+                                                trainer.method_args.retain_loss_type=NLL \
+                                                trainer.method_args.sam_rho=${sam_rho} \
+                                                trainer.method_args.sam_adaptive=${sam_adaptive} \
+                                                trainer.method_args.sam_eps=${sam_eps} \
+                                                retain_logs_path=null \
+                                                "${extra_train_args[@]}" \
+                                                paths.output_dir=${run_dir}
+                                        fi
+
+                                        mkdir -p "${eval_dir}"
+                                        if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                                            rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
+                                        fi
+
+                                        eval_cmd=( \
+                                            experiment=eval/duet/default.yaml \
+                                            model=${lora_model} \
+                                            forget_split=${forget_split} \
+                                            holdout_split=${retain_split} \
+                                            task_name=${task_name} \
+                                            model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                            model.model_args.base_model_name_or_path=${base_model_path} \
+                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                            model.model_args.device_map="auto" \
+                                            model.model_args.low_cpu_mem_usage=true \
+                                            model.lora_config.r=${lora_r} \
+                                            model.lora_config.lora_alpha=${lora_alpha} \
+                                            model.lora_config.lora_dropout=${lora_dropout} \
+                                            eval.duet.overwrite=true \
+                                            "${extra_eval_args[@]}" \
+                                            paths.output_dir=${eval_dir} \
+                                            retain_logs_path=null \
+                                        )
+                                        python src/eval.py "${eval_cmd[@]}"
+
+                                        if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                            if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                                rm -f "${run_dir}"/*.safetensors
+                                                echo "[duet][NPOSAM] Removed safetensors from ${run_dir}"
+                                            fi
+                                        fi
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
diff --git a/scripts/popqa/npo_sam_popqa.sh b/scripts/popqa/npo_sam_popqa.sh
new file mode 100755
index 0000000..03fe4dc
--- /dev/null
+++ b/scripts/popqa/npo_sam_popqa.sh
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
+    echo "[popqa][NPOSAM] Using locally finetuned base checkpoint at ${base_model_path}"
+else
+    base_model_path="${hf_base_model_path}"
+    default_tokenizer_model_path="${hf_base_model_path}"
+    default_tokenizer_subfolder=""
+    echo "[popqa][NPOSAM] Using Hugging Face base checkpoint ${base_model_path}"
+fi
+tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${default_tokenizer_model_path}}"
+tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${default_tokenizer_subfolder}}"
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
+experiment="unlearn/popqa/npo_sam_lora.yaml"
+trainer="NPOSAM"
+
+output_root="${repo_root}/saves/unlearn/popqa/npo_sam"
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
+raw_lrs="${LRS:-1e-5}"
+raw_lrs="${raw_lrs//,/ }"
+raw_lrs="${raw_lrs//\"/}"
+raw_lrs="${raw_lrs//\'/}"
+read -r -a lrs <<< "${raw_lrs}"
+
+raw_betas="${BETAS:-0.5}"
+raw_betas="${raw_betas//,/ }"
+raw_betas="${raw_betas//\"/}"
+raw_betas="${raw_betas//\'/}"
+read -r -a betas <<< "${raw_betas}"
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
+raw_sam_rhos="${SAM_RHOS:-0.01}"
+raw_sam_rhos="${raw_sam_rhos//,/ }"
+raw_sam_rhos="${raw_sam_rhos//\"/}"
+raw_sam_rhos="${raw_sam_rhos//\'/}"
+read -r -a sam_rhos <<< "${raw_sam_rhos}"
+
+raw_sam_adaptives="${SAM_ADAPTIVES:-false}"
+raw_sam_adaptives="${raw_sam_adaptives//,/ }"
+raw_sam_adaptives="${raw_sam_adaptives//\"/}"
+raw_sam_adaptives="${raw_sam_adaptives//\'/}"
+read -r -a sam_adaptives <<< "${raw_sam_adaptives}"
+
+sam_eps="${SAM_EPS:-1e-12}"
+
+lora_rs=(${LORA_RS:-"32"})
+lora_alphas=(${LORA_ALPHAS:-"64"})
+lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
+delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
+
+export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
+
+for split in "${forget_retain_splits[@]}"; do
+    forget_split=$(echo "$split" | cut -d' ' -f1)
+    retain_split=$(echo "$split" | cut -d' ' -f2)
+
+    for lr in "${lrs[@]}"; do
+        for beta in "${betas[@]}"; do
+            beta_tag=${beta//./p}
+            for alpha in "${alphas[@]}"; do
+                alpha_tag=${alpha//./p}
+                for gamma in "${gammas[@]}"; do
+                    gamma_tag=${gamma//./p}
+                    for sam_rho in "${sam_rhos[@]}"; do
+                        rho_tag=${sam_rho//./p}
+                        for sam_adaptive in "${sam_adaptives[@]}"; do
+                            adapt_tag="${sam_adaptive}"
+                            adapt_tag="${adapt_tag//true/T}"
+                            adapt_tag="${adapt_tag//false/F}"
+                            for lora_r in "${lora_rs[@]}"; do
+                                for lora_alpha in "${lora_alphas[@]}"; do
+                                    for lora_dropout in "${lora_dropouts[@]}"; do
+                                        dropout_tag=${lora_dropout//./p}
+                                        task_name=popqa_${base_model}_${forget_split}_npo_sam_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_beta${beta_tag}_alpha${alpha_tag}_gamma${gamma_tag}_rho${rho_tag}_ad${adapt_tag}
+                                        run_dir=${output_root}/${task_name}
+                                        eval_dir=${run_dir}/evals
+                                        summary_path=${eval_dir}/POPQA_SUMMARY.json
+
+                                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                            echo "[popqa][NPOSAM] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                            continue
+                                        fi
+
+                                        echo "[popqa][NPOSAM] ${task_name}: unlearning ${base_model_path} on ${forget_split}"
+
+                                        adapter_path=${run_dir}/adapter_model.safetensors
+                                        if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                            mkdir -p "${run_dir}"
+                                            python src/train.py --config-name=unlearn.yaml \
+                                                experiment=${experiment} \
+                                                trainer=${trainer} \
+                                                task_name=${task_name} \
+                                                model=${lora_model} \
+                                                forget_split=${forget_split} \
+                                                retain_split=${retain_split} \
+                                                model.model_args.pretrained_model_name_or_path=${base_model_path} \
+                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                                model.model_args.device_map="auto" \
+                                                model.model_args.low_cpu_mem_usage=true \
+                                                model.lora_config.r=${lora_r} \
+                                                model.lora_config.lora_alpha=${lora_alpha} \
+                                                model.lora_config.lora_dropout=${lora_dropout} \
+                                                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                                trainer.args.num_train_epochs=${num_train_epochs} \
+                                                trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                                trainer.args.learning_rate=${lr} \
+                                                trainer.method_args.beta=${beta} \
+                                                trainer.method_args.alpha=${alpha} \
+                                                trainer.method_args.gamma=${gamma} \
+                                                trainer.method_args.retain_loss_type=NLL \
+                                                trainer.method_args.sam_rho=${sam_rho} \
+                                                trainer.method_args.sam_adaptive=${sam_adaptive} \
+                                                trainer.method_args.sam_eps=${sam_eps} \
+                                                retain_logs_path=null \
+                                                "${extra_train_args[@]}" \
+                                                paths.output_dir=${run_dir}
+                                        fi
+
+                                        mkdir -p "${eval_dir}"
+                                        if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
+                                            rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
+                                        fi
+
+                                        eval_cmd=( \
+                                            experiment=eval/popqa/default.yaml \
+                                            model=${lora_model} \
+                                            forget_split=${forget_split} \
+                                            holdout_split=${retain_split} \
+                                            task_name=${task_name} \
+                                            model.model_args.pretrained_model_name_or_path=${run_dir} \
+                                            model.model_args.base_model_name_or_path=${base_model_path} \
+                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                            model.model_args.device_map="auto" \
+                                            model.model_args.low_cpu_mem_usage=true \
+                                            model.lora_config.r=${lora_r} \
+                                            model.lora_config.lora_alpha=${lora_alpha} \
+                                            model.lora_config.lora_dropout=${lora_dropout} \
+                                            eval.duet.overwrite=true \
+                                            "${extra_eval_args[@]}" \
+                                            paths.output_dir=${eval_dir} \
+                                            retain_logs_path=null \
+                                        )
+                                        python src/eval.py "${eval_cmd[@]}"
+
+                                        if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                            if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                                rm -f "${run_dir}"/*.safetensors
+                                                echo "[popqa][NPOSAM] Removed safetensors from ${run_dir}"
+                                            fi
+                                        fi
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
diff --git a/src/trainer/__init__.py b/src/trainer/__init__.py
index aca89de..7f4ccdc 100644
--- a/src/trainer/__init__.py
+++ b/src/trainer/__init__.py
@@ -7,6 +7,7 @@ from trainer.base import FinetuneTrainer
 from trainer.unlearn.grad_ascent import GradAscent
 from trainer.unlearn.grad_diff import GradDiff
 from trainer.unlearn.npo import NPO
+from trainer.unlearn.npo_sam import NPOSAM
 from trainer.unlearn.dpo import DPO
 from trainer.unlearn.simnpo import SimNPO
 from trainer.unlearn.rmu import RMU
@@ -95,6 +96,7 @@ _register_trainer(FinetuneTrainer)
 _register_trainer(GradAscent)
 _register_trainer(GradDiff)
 _register_trainer(NPO)
+_register_trainer(NPOSAM)
 _register_trainer(DPO)
 _register_trainer(SimNPO)
 _register_trainer(RMU)
diff --git a/src/trainer/unlearn/npo_sam.py b/src/trainer/unlearn/npo_sam.py
new file mode 100644
index 0000000..d867483
--- /dev/null
+++ b/src/trainer/unlearn/npo_sam.py
@@ -0,0 +1,260 @@
+from __future__ import annotations
+
+from dataclasses import dataclass
+from typing import List, Optional, Sequence
+
+import torch
+
+from trainer.unlearn.npo import NPO
+from trainer.utils import compute_dpo_loss
+
+
+@dataclass
+class _SAMState:
+    e_ws: List[Optional[torch.Tensor]]
+
+
+class NPOSAM(NPO):
+    """
+    NPO + SAM (Sharpness-Aware Minimization), as used in Unlearn-Smooth.
+
+    SAM is implemented with a two-pass update:
+    1) compute gradients at current weights, then perturb parameters by +e(w)
+    2) compute gradients at perturbed weights (actual update gradients)
+    3) restore original weights
+    """
+
+    def __init__(
+        self,
+        sam_rho: float = 0.01,
+        sam_adaptive: bool = False,
+        sam_eps: float = 1e-12,
+        *args,
+        **kwargs,
+    ):
+        super().__init__(*args, **kwargs)
+        self.sam_rho = float(sam_rho)
+        self.sam_adaptive = bool(sam_adaptive)
+        self.sam_eps = float(sam_eps)
+
+    def _trainable_params(self, model: torch.nn.Module) -> List[torch.nn.Parameter]:
+        return [p for p in model.parameters() if p.requires_grad]
+
+    def _stash_grads(self, params: List[torch.nn.Parameter]) -> List[Optional[torch.Tensor]]:
+        stashed: List[Optional[torch.Tensor]] = []
+        for p in params:
+            if p.grad is None:
+                stashed.append(None)
+            else:
+                stashed.append(p.grad.detach().clone())
+        return stashed
+
+    def _clear_grads_set_to_none(self, params: List[torch.nn.Parameter]) -> None:
+        for p in params:
+            p.grad = None
+
+    @torch.no_grad()
+    def _grad_norm(
+        self,
+        params: List[torch.nn.Parameter],
+        grads: Sequence[Optional[torch.Tensor]],
+    ) -> torch.Tensor:
+        if not params:
+            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
+
+        ref_device = None
+        for g in grads:
+            if g is not None:
+                ref_device = g.device
+                break
+        if ref_device is None:
+            return torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
+
+        sq_sum = torch.zeros((), device=ref_device, dtype=torch.float32)
+        for p, g in zip(params, grads):
+            if g is None:
+                continue
+            grad = g
+            if self.sam_adaptive:
+                grad = p.detach().abs() * grad
+            grad_sq = grad.float()
+            if grad_sq.device != ref_device:
+                grad_sq = grad_sq.to(ref_device)
+            sq_sum = sq_sum + (grad_sq * grad_sq).sum()
+        return torch.sqrt(sq_sum)
+
+    @torch.no_grad()
+    def _perturb_weights(
+        self,
+        params: List[torch.nn.Parameter],
+        grads: Sequence[Optional[torch.Tensor]],
+        grad_norm: torch.Tensor,
+    ) -> _SAMState:
+        scale = self.sam_rho / (grad_norm + self.sam_eps)
+        e_ws: List[Optional[torch.Tensor]] = []
+
+        for p, g in zip(params, grads):
+            if g is None:
+                e_ws.append(None)
+                continue
+
+            if self.sam_adaptive:
+                perturb = p.detach().abs() * g
+            else:
+                perturb = g
+
+            scale_t = scale.to(device=perturb.device, dtype=perturb.dtype)
+            e_w = (perturb * scale_t).to(dtype=p.dtype)
+            p.add_(e_w)
+            e_ws.append(e_w)
+
+        return _SAMState(e_ws=e_ws)
+
+    @torch.no_grad()
+    def _restore_weights(
+        self, params: List[torch.nn.Parameter], state: _SAMState
+    ) -> None:
+        for p, e_w in zip(params, state.e_ws):
+            if e_w is None:
+                continue
+            p.sub_(e_w)
+
+    def _set_final_grads(
+        self,
+        params: List[torch.nn.Parameter],
+        second_pass_grads: Sequence[Optional[torch.Tensor]],
+        prev_grads: List[Optional[torch.Tensor]],
+        grad_scale: float,
+    ) -> None:
+        for p, g2, g_prev in zip(params, second_pass_grads, prev_grads):
+            grad = None
+            if g2 is not None:
+                grad = g2.detach() * grad_scale
+
+            if g_prev is not None:
+                if grad is None:
+                    grad = g_prev
+                else:
+                    grad = grad + g_prev
+
+            p.grad = grad
+
+    def _compute_forget_loss_only(self, model, inputs):
+        forget_inputs = inputs["forget"]
+        if isinstance(forget_inputs, dict) and "original" in forget_inputs:
+            forget_inputs = forget_inputs["original"]
+
+        forget_loss, _ = compute_dpo_loss(
+            model=model,
+            ref_model=self.ref_model,
+            win_inputs=None,
+            lose_inputs=forget_inputs,
+            beta=self.beta,
+        )
+        return forget_loss
+
+    def _compute_retain_loss_only(self, model, inputs):
+        retain_inputs = inputs["retain"]
+        retain_inputs = {
+            "input_ids": retain_inputs["input_ids"],
+            "attention_mask": retain_inputs["attention_mask"],
+            "labels": retain_inputs["labels"],
+        }
+        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
+
+    def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
+        if self.is_deepspeed_enabled:
+            raise NotImplementedError(
+                "[NPOSAM] DeepSpeed is not supported in this integration."
+            )
+        if getattr(self.accelerator, "num_processes", 1) > 1:
+            raise NotImplementedError(
+                "[NPOSAM] Multi-process training is not supported in this integration."
+            )
+
+        model.train()
+        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
+            self.optimizer.train()
+
+        inputs = self._prepare_inputs(inputs)
+        params = self._trainable_params(model)
+        if not params:
+            raise RuntimeError("[NPOSAM] No trainable parameters found.")
+
+        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
+        grad_scale = 1.0 / grad_acc_steps
+
+        prev_grads = self._stash_grads(params)
+        self._clear_grads_set_to_none(params)
+
+        # 1) Forget pass at current weights (for SAM perturbation direction).
+        with self.compute_loss_context_manager():
+            forget_loss_1 = self._compute_forget_loss_only(model, inputs)
+        grads_1 = torch.autograd.grad(
+            forget_loss_1,
+            params,
+            retain_graph=False,
+            create_graph=False,
+            allow_unused=True,
+        )
+
+        grad_norm = self._grad_norm(params, grads_1)
+        sam_state = self._perturb_weights(params, grads_1, grad_norm)
+
+        # 2) Forget pass at perturbed weights (SAM second pass).
+        try:
+            self._clear_grads_set_to_none(params)
+            with self.compute_loss_context_manager():
+                forget_loss_2 = self._compute_forget_loss_only(model, inputs)
+            forget_grads = torch.autograd.grad(
+                forget_loss_2,
+                params,
+                retain_graph=False,
+                create_graph=False,
+                allow_unused=True,
+            )
+        finally:
+            # Restore weights even if an exception occurs.
+            self._restore_weights(params, sam_state)
+
+        # 3) Retain gradient at restored (unperturbed) weights.
+        self._clear_grads_set_to_none(params)
+        with self.compute_loss_context_manager():
+            retain_loss = self._compute_retain_loss_only(model, inputs)
+        retain_grads = torch.autograd.grad(
+            retain_loss,
+            params,
+            retain_graph=False,
+            create_graph=False,
+            allow_unused=True,
+        )
+
+        # 4) Combine forget and retain gradients with NPO weights.
+        combined_grads: List[Optional[torch.Tensor]] = []
+        for g_forget, g_retain in zip(forget_grads, retain_grads):
+            grad = None
+            if g_forget is not None:
+                grad = self.gamma * g_forget
+            if g_retain is not None:
+                retain_component = self.alpha * g_retain
+                grad = retain_component if grad is None else grad + retain_component
+            combined_grads.append(grad)
+
+        self._set_final_grads(params, combined_grads, prev_grads, grad_scale)
+
+        try:
+            self.log(
+                {
+                    "npo_sam_forget_loss_1": float(forget_loss_1.detach().item()),
+                    "npo_sam_forget_loss_2": float(forget_loss_2.detach().item()),
+                    "npo_sam_retain_loss": float(retain_loss.detach().item()),
+                    "npo_sam_grad_norm": float(grad_norm.detach().item()),
+                    "npo_sam_rho": float(self.sam_rho),
+                    "npo_sam_adaptive": 1.0 if self.sam_adaptive else 0.0,
+                }
+            )
+        except Exception:
+            pass
+
+        total_loss = self.gamma * forget_loss_2 + self.alpha * retain_loss
+        return total_loss.detach() * grad_scale
```

## 2026-03-04 DUET Grid Search Defaults Update

Updated script:
- `scripts/duet/npo_sam_duet.sh`

New default grid values:
- `LRS`: `1e-5`
- `BETAS`: `0.1 0.015`
- `SAM_RHOS`: `0.01`
- `ALPHAS`: `1.0`
- `GAMMAS`: `1.0 2.25`

Notes:
- `SAM_RHOS` already matched the target (`0.01`) and was left unchanged.

## 2026-03-04 DUET LR Default Update

Updated script:
- `scripts/duet/npo_sam_duet.sh`

New default:
- `LRS`: `1e-3`

## 2026-03-05 RWKU NPOSAM Integration

Added files:
- `configs/experiment/unlearn/rwku/npo_sam_lora.yaml`
- `scripts/rwku/npo_sam_rwku.sh`

RWKU script defaults now mirror DUET NPOSAM defaults (while keeping RWKU dataset + default base Llama):
- `BASE_MODEL`: `Llama-3.1-8B`
- `LRS`: `1e-3`
- `BETAS`: `0.1`
- `ALPHAS`: `1.0`
- `GAMMAS`: `1.0`
- `SAM_RHOS`: `0.01`
- `SAM_ADAPTIVES`: `false`
- `SAM_EPS`: `1e-12`
- `FORGET_SPLIT`: `forget_level2`
- `RETAIN_SPLIT`: `neighbor_level2`
- `EVAL_BATCH_SIZE`: `8` (override supported, same knob as DUET script)
- `DELETE_MODEL_SAFETENSORS_AFTER_EVAL`: supported

Notes:
- Added optional `HF_TOKEN` -> `HUGGINGFACE_HUB_TOKEN` mapping for gated model access without hardcoding secrets.

## 2026-03-12 Current Production Baseline Update

Updated files:
- `configs/experiment/unlearn/duet/npo_sam_lora.yaml`
- `configs/experiment/unlearn/rwku/npo_sam_lora.yaml`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/rwku/npo_sam_rwku.sh`
- `configs/model/Llama-3.1-8B-Instruct-lora.yaml`
- `prod-gpu-runs-new.md`

What changed:
- Active NPOSAM runs now default to `NUM_EPOCHS=2`.
- Active NPOSAM scripts now default to `LRS="1e-6 5e-6 1e-5 5e-5 1e-4"`.
- `GRADIENT_CHECKPOINTING` remains default `false` in both DUET and RWKU NPOSAM scripts/configs.
- Default LoRA target modules were reduced to attention-only adapters: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- RWKU NPOSAM moved to `Llama-3.1-8B-Instruct` for the current production run stack.

## 2026-03-12 Qwen/Gemma LoRA Alignment

Updated files:
- `configs/model/Qwen2.5-7B-Instruct-lora.yaml`
- `configs/model/gemma-7b-it-lora.yaml`

What changed:
- Qwen2.5-7B-Instruct and gemma-7b-it LoRA configs were aligned with the active attention-only adapter policy.
- Default target modules are now `q_proj`, `k_proj`, `v_proj`, `o_proj`.
