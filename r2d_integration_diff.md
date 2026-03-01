# R2D Integration Diff

Base commit: a85c7f3509a9f6a5836fa3e63b0c385609f79cbc (before sweep-script updates)
Target: current working tree

```diff
diff --git a/scripts/duet/r2d_duet.sh b/scripts/duet/r2d_duet.sh
index 2de4604..a641baf 100755
--- a/scripts/duet/r2d_duet.sh
+++ b/scripts/duet/r2d_duet.sh
@@ -74,7 +74,6 @@ per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
 gradient_accumulation_steps=${GRAD_ACCUM:-32}
 eval_batch_size=${EVAL_BATCH_SIZE:-8}
 num_train_epochs=${NUM_EPOCHS:-1}
-max_steps=${R2D_MAX_STEPS:-0}
 gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
 
 raw_lrs="${LRS:-1e-5}"
@@ -87,9 +86,8 @@ lora_rs=(${LORA_RS:-"32"})
 lora_alphas=(${LORA_ALPHAS:-"64"})
 lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
 
-r2d_noise_std="${R2D_NOISE_STD:-null}"
+r2d_noise_trainable_only="${R2D_NOISE_TRAINABLE_ONLY:-true}"
 r2d_noise_seed="${R2D_NOISE_SEED:-0}"
-r2d_eps="${R2D_EPS:-null}"
 r2d_delta="${R2D_DELTA:-null}"
 r2d_sens="${R2D_SENS:-null}"
 r2d_use_analytic_gaussian="${R2D_USE_ANALYTIC_GAUSSIAN:-true}"
@@ -113,7 +111,25 @@ if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
 fi
 delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
 
-if [[ "${max_steps}" == "0" ]]; then
+raw_max_steps="${R2D_MAX_STEPS_LIST:-${R2D_MAX_STEPS:-0}}"
+raw_max_steps="${raw_max_steps//,/ }"
+raw_max_steps="${raw_max_steps//\"/}"
+raw_max_steps="${raw_max_steps//\'/}"
+read -r -a max_steps_list <<< "${raw_max_steps}"
+
+raw_noise_stds="${R2D_NOISE_STDS:-${R2D_NOISE_STD:-null}}"
+raw_noise_stds="${raw_noise_stds//,/ }"
+raw_noise_stds="${raw_noise_stds//\"/}"
+raw_noise_stds="${raw_noise_stds//\'/}"
+read -r -a noise_stds <<< "${raw_noise_stds}"
+
+raw_epsilons="${R2D_EPSILONS:-${R2D_EPS:-null}}"
+raw_epsilons="${raw_epsilons//,/ }"
+raw_epsilons="${raw_epsilons//\"/}"
+raw_epsilons="${raw_epsilons//\'/}"
+read -r -a epsilons <<< "${raw_epsilons}"
+
+if [[ "${#max_steps_list[@]}" -eq 1 && "${max_steps_list[0]}" == "0" ]]; then
     echo "[duet][R2D] WARNING: R2D_MAX_STEPS=0, using NUM_EPOCHS instead of explicit K steps."
 fi
 
@@ -125,119 +141,151 @@ for split in "${forget_retain_splits[@]}"; do
         forget_label="${forget_split}"
     fi
 
-    for lr in "${lrs[@]}"; do
-        for lora_r in "${lora_rs[@]}"; do
-            for lora_alpha in "${lora_alphas[@]}"; do
-                for lora_dropout in "${lora_dropouts[@]}"; do
-                    dropout_tag=${lora_dropout//./p}
-                    rewind_tag="${R2D_REWIND_TAG:-rewind}"
-                    sigma_tag=${r2d_noise_std//./p}
-                    sigma_tag=${sigma_tag//null/dp}
-                    task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
-                    run_dir=${output_root}/${task_name}
-                    eval_dir=${run_dir}/evals
-                    summary_path=${eval_dir}/DUET_SUMMARY.json
-
-                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
-                        echo "[duet][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
-                        continue
-                    fi
+    for max_steps in "${max_steps_list[@]}"; do
+        if [[ "${max_steps}" == "0" ]]; then
+            k_tag="ep${num_train_epochs}"
+        else
+            k_tag="k${max_steps}"
+        fi
 
-                    echo "[duet][R2D] ${task_name}"
-                    echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
-                    echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps}"
-
-                    adapter_path=${run_dir}/adapter_model.safetensors
-                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
-                        mkdir -p "${run_dir}"
-
-                        train_cmd=( \
-                            --config-name=unlearn.yaml \
-                            experiment=${experiment} \
-                            trainer=${trainer} \
-                            task_name=${task_name} \
-                            model=${lora_model} \
-                            forget_split=${forget_split} \
-                            retain_split=${retain_split} \
-                            holdout_split=${retain_split} \
-                            model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
-                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
-                            model.model_args.device_map="auto" \
-                            model.model_args.low_cpu_mem_usage=true \
-                            model.lora_config.r=${lora_r} \
-                            model.lora_config.lora_alpha=${lora_alpha} \
-                            model.lora_config.lora_dropout=${lora_dropout} \
-                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
-                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
-                            trainer.args.num_train_epochs=${num_train_epochs} \
-                            trainer.args.gradient_checkpointing=${gradient_checkpointing} \
-                            trainer.args.learning_rate=${lr} \
-                            trainer.args.optim=sgd \
-                            trainer.args.weight_decay=0.0 \
-                            trainer.args.lr_scheduler_type=constant \
-                            trainer.args.warmup_ratio=0.0 \
-                            trainer.args.save_strategy=no \
-                            trainer.args.eval_strategy=no \
-                            trainer.args.do_eval=false \
-                            trainer.args.eval_on_start=false \
-                            trainer.args.report_to=none \
-                            trainer.method_args.noise_std=${r2d_noise_std} \
-                            trainer.method_args.noise_seed=${r2d_noise_seed} \
-                            trainer.method_args.noise_trainable_only=true \
-                            trainer.method_args.dp_epsilon=${r2d_eps} \
-                            trainer.method_args.dp_delta=${r2d_delta} \
-                            trainer.method_args.dp_sensitivity=${r2d_sens} \
-                            trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
-                            trainer.method_args.r2d_L=${r2d_L} \
-                            trainer.method_args.r2d_G=${r2d_G} \
-                            trainer.method_args.r2d_n=${r2d_n} \
-                            trainer.method_args.r2d_m=${r2d_m} \
-                            trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
-                            trainer.method_args.r2d_eta=${r2d_eta} \
-                            retain_logs_path=null \
-                            "${extra_train_args[@]}" \
-                            paths.output_dir=${run_dir} \
-                        )
-                        if [[ "${max_steps}" != "0" ]]; then
-                            train_cmd+=(trainer.args.max_steps=${max_steps})
-                        fi
-
-                        python src/train.py "${train_cmd[@]}"
-                    fi
+        for r2d_noise_std in "${noise_stds[@]}"; do
+            eps_candidates=("null")
+            if [[ "${r2d_noise_std}" == "null" ]]; then
+                eps_candidates=("${epsilons[@]}")
+            fi
 
-                    mkdir -p "${eval_dir}"
-                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
-                        rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
+            for r2d_eps in "${eps_candidates[@]}"; do
+                if [[ "${r2d_noise_std}" != "null" ]]; then
+                    sigma_tag="s${r2d_noise_std//./p}"
+                    noise_mode="direct"
+                else
+                    eps_tag=${r2d_eps//./p}
+                    eps_tag=${eps_tag//null/none}
+                    delta_tag=${r2d_delta//./p}
+                    delta_tag=${delta_tag//null/none}
+                    sigma_tag="dp_e${eps_tag}_d${delta_tag}"
+                    if [[ "${r2d_sens}" != "null" ]]; then
+                        sigma_tag="${sigma_tag}_gs${r2d_sens//./p}"
+                    elif [[ "${r2d_rewind_step_for_sigma}" != "null" ]]; then
+                        sigma_tag="${sigma_tag}_rw${r2d_rewind_step_for_sigma}"
                     fi
+                    noise_mode="dp"
+                fi
 
-                    eval_cmd=( \
-                        experiment=eval/duet/default.yaml \
-                        model=${lora_model} \
-                        forget_split=${forget_split} \
-                        holdout_split=${retain_split} \
-                        task_name=${task_name} \
-                        model.model_args.pretrained_model_name_or_path=${run_dir} \
-                        model.model_args.base_model_name_or_path=${rewind_model_path} \
-                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
-                        model.model_args.device_map="auto" \
-                        model.model_args.low_cpu_mem_usage=true \
-                        model.lora_config.r=${lora_r} \
-                        model.lora_config.lora_alpha=${lora_alpha} \
-                        model.lora_config.lora_dropout=${lora_dropout} \
-                        eval.duet.batch_size=${eval_batch_size} \
-                        eval.duet.overwrite=true \
-                        "${extra_eval_args[@]}" \
-                        paths.output_dir=${eval_dir} \
-                        retain_logs_path=null \
-                    )
-                    python src/eval.py "${eval_cmd[@]}"
-
-                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
-                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
-                            rm -f "${run_dir}"/*.safetensors
-                            echo "[duet][R2D] Removed safetensors from ${run_dir}"
-                        fi
-                    fi
+                for lr in "${lrs[@]}"; do
+                    for lora_r in "${lora_rs[@]}"; do
+                        for lora_alpha in "${lora_alphas[@]}"; do
+                            for lora_dropout in "${lora_dropouts[@]}"; do
+                                dropout_tag=${lora_dropout//./p}
+                                rewind_tag="${R2D_REWIND_TAG:-rewind}"
+                                task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_${k_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                                run_dir=${output_root}/${task_name}
+                                eval_dir=${run_dir}/evals
+                                summary_path=${eval_dir}/DUET_SUMMARY.json
+
+                                if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                    echo "[duet][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                    continue
+                                fi
+
+                                echo "[duet][R2D] ${task_name}"
+                                echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
+                                echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps} mode=${noise_mode} eps=${r2d_eps}"
+
+                                adapter_path=${run_dir}/adapter_model.safetensors
+                                if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                    mkdir -p "${run_dir}"
+
+                                    train_cmd=( \
+                                        --config-name=unlearn.yaml \
+                                        experiment=${experiment} \
+                                        trainer=${trainer} \
+                                        task_name=${task_name} \
+                                        model=${lora_model} \
+                                        forget_split=${forget_split} \
+                                        retain_split=${retain_split} \
+                                        holdout_split=${retain_split} \
+                                        model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
+                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                        model.model_args.device_map="auto" \
+                                        model.model_args.low_cpu_mem_usage=true \
+                                        model.lora_config.r=${lora_r} \
+                                        model.lora_config.lora_alpha=${lora_alpha} \
+                                        model.lora_config.lora_dropout=${lora_dropout} \
+                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                        trainer.args.num_train_epochs=${num_train_epochs} \
+                                        trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                        trainer.args.learning_rate=${lr} \
+                                        trainer.args.optim=sgd \
+                                        trainer.args.weight_decay=0.0 \
+                                        trainer.args.lr_scheduler_type=constant \
+                                        trainer.args.warmup_ratio=0.0 \
+                                        trainer.args.save_strategy=no \
+                                        trainer.args.eval_strategy=no \
+                                        trainer.args.do_eval=false \
+                                        trainer.args.eval_on_start=false \
+                                        trainer.args.report_to=none \
+                                        trainer.method_args.noise_std=${r2d_noise_std} \
+                                        trainer.method_args.noise_seed=${r2d_noise_seed} \
+                                        trainer.method_args.noise_trainable_only=${r2d_noise_trainable_only} \
+                                        trainer.method_args.dp_epsilon=${r2d_eps} \
+                                        trainer.method_args.dp_delta=${r2d_delta} \
+                                        trainer.method_args.dp_sensitivity=${r2d_sens} \
+                                        trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
+                                        trainer.method_args.r2d_L=${r2d_L} \
+                                        trainer.method_args.r2d_G=${r2d_G} \
+                                        trainer.method_args.r2d_n=${r2d_n} \
+                                        trainer.method_args.r2d_m=${r2d_m} \
+                                        trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
+                                        trainer.method_args.r2d_eta=${r2d_eta} \
+                                        retain_logs_path=null \
+                                        "${extra_train_args[@]}" \
+                                        paths.output_dir=${run_dir} \
+                                    )
+                                    if [[ "${max_steps}" != "0" ]]; then
+                                        train_cmd+=(trainer.args.max_steps=${max_steps})
+                                    fi
+
+                                    python src/train.py "${train_cmd[@]}"
+                                fi
+
+                                mkdir -p "${eval_dir}"
+                                if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
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
+                                    model.model_args.base_model_name_or_path=${rewind_model_path} \
+                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                    model.model_args.device_map="auto" \
+                                    model.model_args.low_cpu_mem_usage=true \
+                                    model.lora_config.r=${lora_r} \
+                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                    eval.duet.batch_size=${eval_batch_size} \
+                                    eval.duet.overwrite=true \
+                                    "${extra_eval_args[@]}" \
+                                    paths.output_dir=${eval_dir} \
+                                    retain_logs_path=null \
+                                )
+                                python src/eval.py "${eval_cmd[@]}"
+
+                                if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                    if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                        rm -f "${run_dir}"/*.safetensors
+                                        echo "[duet][R2D] Removed safetensors from ${run_dir}"
+                                    fi
+                                fi
+                            done
+                        done
+                    done
                 done
             done
         done
diff --git a/scripts/popqa/r2d_popqa.sh b/scripts/popqa/r2d_popqa.sh
index 9f5c054..c5c66ec 100755
--- a/scripts/popqa/r2d_popqa.sh
+++ b/scripts/popqa/r2d_popqa.sh
@@ -85,7 +85,6 @@ per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
 gradient_accumulation_steps=${GRAD_ACCUM:-32}
 eval_batch_size=${EVAL_BATCH_SIZE:-8}
 num_train_epochs=${NUM_EPOCHS:-1}
-max_steps=${R2D_MAX_STEPS:-0}
 gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
 
 raw_lrs="${LRS:-1e-5}"
@@ -98,9 +97,8 @@ lora_rs=(${LORA_RS:-"32"})
 lora_alphas=(${LORA_ALPHAS:-"64"})
 lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
 
-r2d_noise_std="${R2D_NOISE_STD:-null}"
+r2d_noise_trainable_only="${R2D_NOISE_TRAINABLE_ONLY:-true}"
 r2d_noise_seed="${R2D_NOISE_SEED:-0}"
-r2d_eps="${R2D_EPS:-null}"
 r2d_delta="${R2D_DELTA:-null}"
 r2d_sens="${R2D_SENS:-null}"
 r2d_use_analytic_gaussian="${R2D_USE_ANALYTIC_GAUSSIAN:-true}"
@@ -124,7 +122,25 @@ if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
 fi
 delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
 
-if [[ "${max_steps}" == "0" ]]; then
+raw_max_steps="${R2D_MAX_STEPS_LIST:-${R2D_MAX_STEPS:-0}}"
+raw_max_steps="${raw_max_steps//,/ }"
+raw_max_steps="${raw_max_steps//\"/}"
+raw_max_steps="${raw_max_steps//\'/}"
+read -r -a max_steps_list <<< "${raw_max_steps}"
+
+raw_noise_stds="${R2D_NOISE_STDS:-${R2D_NOISE_STD:-null}}"
+raw_noise_stds="${raw_noise_stds//,/ }"
+raw_noise_stds="${raw_noise_stds//\"/}"
+raw_noise_stds="${raw_noise_stds//\'/}"
+read -r -a noise_stds <<< "${raw_noise_stds}"
+
+raw_epsilons="${R2D_EPSILONS:-${R2D_EPS:-null}}"
+raw_epsilons="${raw_epsilons//,/ }"
+raw_epsilons="${raw_epsilons//\"/}"
+raw_epsilons="${raw_epsilons//\'/}"
+read -r -a epsilons <<< "${raw_epsilons}"
+
+if [[ "${#max_steps_list[@]}" -eq 1 && "${max_steps_list[0]}" == "0" ]]; then
     echo "[popqa][R2D] WARNING: R2D_MAX_STEPS=0, using NUM_EPOCHS instead of explicit K steps."
 fi
 
@@ -136,119 +152,151 @@ for split in "${forget_retain_splits[@]}"; do
         forget_label="${forget_split}"
     fi
 
-    for lr in "${lrs[@]}"; do
-        for lora_r in "${lora_rs[@]}"; do
-            for lora_alpha in "${lora_alphas[@]}"; do
-                for lora_dropout in "${lora_dropouts[@]}"; do
-                    dropout_tag=${lora_dropout//./p}
-                    rewind_tag="${R2D_REWIND_TAG:-rewind}"
-                    sigma_tag=${r2d_noise_std//./p}
-                    sigma_tag=${sigma_tag//null/dp}
-                    task_name=popqa_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
-                    run_dir=${output_root}/${task_name}
-                    eval_dir=${run_dir}/evals
-                    summary_path=${eval_dir}/POPQA_SUMMARY.json
-
-                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
-                        echo "[popqa][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
-                        continue
-                    fi
-
-                    echo "[popqa][R2D] ${task_name}"
-                    echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
-                    echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps}"
-
-                    adapter_path=${run_dir}/adapter_model.safetensors
-                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
-                        mkdir -p "${run_dir}"
-
-                        train_cmd=( \
-                            --config-name=unlearn.yaml \
-                            experiment=${experiment} \
-                            trainer=${trainer} \
-                            task_name=${task_name} \
-                            model=${lora_model} \
-                            forget_split=${forget_split} \
-                            retain_split=${retain_split} \
-                            holdout_split=${retain_split} \
-                            model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
-                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
-                            model.model_args.device_map="auto" \
-                            model.model_args.low_cpu_mem_usage=true \
-                            model.lora_config.r=${lora_r} \
-                            model.lora_config.lora_alpha=${lora_alpha} \
-                            model.lora_config.lora_dropout=${lora_dropout} \
-                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
-                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
-                            trainer.args.num_train_epochs=${num_train_epochs} \
-                            trainer.args.gradient_checkpointing=${gradient_checkpointing} \
-                            trainer.args.learning_rate=${lr} \
-                            trainer.args.optim=sgd \
-                            trainer.args.weight_decay=0.0 \
-                            trainer.args.lr_scheduler_type=constant \
-                            trainer.args.warmup_ratio=0.0 \
-                            trainer.args.save_strategy=no \
-                            trainer.args.eval_strategy=no \
-                            trainer.args.do_eval=false \
-                            trainer.args.eval_on_start=false \
-                            trainer.args.report_to=none \
-                            trainer.method_args.noise_std=${r2d_noise_std} \
-                            trainer.method_args.noise_seed=${r2d_noise_seed} \
-                            trainer.method_args.noise_trainable_only=true \
-                            trainer.method_args.dp_epsilon=${r2d_eps} \
-                            trainer.method_args.dp_delta=${r2d_delta} \
-                            trainer.method_args.dp_sensitivity=${r2d_sens} \
-                            trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
-                            trainer.method_args.r2d_L=${r2d_L} \
-                            trainer.method_args.r2d_G=${r2d_G} \
-                            trainer.method_args.r2d_n=${r2d_n} \
-                            trainer.method_args.r2d_m=${r2d_m} \
-                            trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
-                            trainer.method_args.r2d_eta=${r2d_eta} \
-                            retain_logs_path=null \
-                            "${extra_train_args[@]}" \
-                            paths.output_dir=${run_dir} \
-                        )
-                        if [[ "${max_steps}" != "0" ]]; then
-                            train_cmd+=(trainer.args.max_steps=${max_steps})
-                        fi
-
-                        python src/train.py "${train_cmd[@]}"
-                    fi
-
-                    mkdir -p "${eval_dir}"
-                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
-                        rm -f "${summary_path}" "${eval_dir}/POPQA_EVAL.json"
-                    fi
-
-                    eval_cmd=( \
-                        experiment=eval/popqa/default.yaml \
-                        model=${lora_model} \
-                        forget_split=${forget_split} \
-                        holdout_split=${retain_split} \
-                        task_name=${task_name} \
-                        model.model_args.pretrained_model_name_or_path=${run_dir} \
-                        model.model_args.base_model_name_or_path=${rewind_model_path} \
-                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
-                        model.model_args.device_map="auto" \
-                        model.model_args.low_cpu_mem_usage=true \
-                        model.lora_config.r=${lora_r} \
-                        model.lora_config.lora_alpha=${lora_alpha} \
-                        model.lora_config.lora_dropout=${lora_dropout} \
-                        eval.duet.batch_size=${eval_batch_size} \
-                        eval.duet.overwrite=true \
-                        "${extra_eval_args[@]}" \
-                        paths.output_dir=${eval_dir} \
-                        retain_logs_path=null \
-                    )
-                    python src/eval.py "${eval_cmd[@]}"
-
-                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
-                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
-                            rm -f "${run_dir}"/*.safetensors
-                            echo "[popqa][R2D] Removed safetensors from ${run_dir}"
-                        fi
+    for max_steps in "${max_steps_list[@]}"; do
+        if [[ "${max_steps}" == "0" ]]; then
+            k_tag="ep${num_train_epochs}"
+        else
+            k_tag="k${max_steps}"
+        fi
+
+        for r2d_noise_std in "${noise_stds[@]}"; do
+            eps_candidates=("null")
+            if [[ "${r2d_noise_std}" == "null" ]]; then
+                eps_candidates=("${epsilons[@]}")
+            fi
+
+            for r2d_eps in "${eps_candidates[@]}"; do
+                if [[ "${r2d_noise_std}" != "null" ]]; then
+                    sigma_tag="s${r2d_noise_std//./p}"
+                    noise_mode="direct"
+                else
+                    eps_tag=${r2d_eps//./p}
+                    eps_tag=${eps_tag//null/none}
+                    delta_tag=${r2d_delta//./p}
+                    delta_tag=${delta_tag//null/none}
+                    sigma_tag="dp_e${eps_tag}_d${delta_tag}"
+                    if [[ "${r2d_sens}" != "null" ]]; then
+                        sigma_tag="${sigma_tag}_gs${r2d_sens//./p}"
+                    elif [[ "${r2d_rewind_step_for_sigma}" != "null" ]]; then
+                        sigma_tag="${sigma_tag}_rw${r2d_rewind_step_for_sigma}"
                     fi
+                    noise_mode="dp"
+                fi
+
+                for lr in "${lrs[@]}"; do
+                    for lora_r in "${lora_rs[@]}"; do
+                        for lora_alpha in "${lora_alphas[@]}"; do
+                            for lora_dropout in "${lora_dropouts[@]}"; do
+                                dropout_tag=${lora_dropout//./p}
+                                rewind_tag="${R2D_REWIND_TAG:-rewind}"
+                                task_name=popqa_${base_model}_${forget_label}_r2d_${rewind_tag}_${k_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                                run_dir=${output_root}/${task_name}
+                                eval_dir=${run_dir}/evals
+                                summary_path=${eval_dir}/POPQA_SUMMARY.json
+
+                                if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
+                                    echo "[popqa][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
+                                    continue
+                                fi
+
+                                echo "[popqa][R2D] ${task_name}"
+                                echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder}"
+                                echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps} mode=${noise_mode} eps=${r2d_eps}"
+
+                                adapter_path=${run_dir}/adapter_model.safetensors
+                                if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
+                                    mkdir -p "${run_dir}"
+
+                                    train_cmd=( \
+                                        --config-name=unlearn.yaml \
+                                        experiment=${experiment} \
+                                        trainer=${trainer} \
+                                        task_name=${task_name} \
+                                        model=${lora_model} \
+                                        forget_split=${forget_split} \
+                                        retain_split=${retain_split} \
+                                        holdout_split=${retain_split} \
+                                        model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
+                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                        model.model_args.device_map="auto" \
+                                        model.model_args.low_cpu_mem_usage=true \
+                                        model.lora_config.r=${lora_r} \
+                                        model.lora_config.lora_alpha=${lora_alpha} \
+                                        model.lora_config.lora_dropout=${lora_dropout} \
+                                        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
+                                        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
+                                        trainer.args.num_train_epochs=${num_train_epochs} \
+                                        trainer.args.gradient_checkpointing=${gradient_checkpointing} \
+                                        trainer.args.learning_rate=${lr} \
+                                        trainer.args.optim=sgd \
+                                        trainer.args.weight_decay=0.0 \
+                                        trainer.args.lr_scheduler_type=constant \
+                                        trainer.args.warmup_ratio=0.0 \
+                                        trainer.args.save_strategy=no \
+                                        trainer.args.eval_strategy=no \
+                                        trainer.args.do_eval=false \
+                                        trainer.args.eval_on_start=false \
+                                        trainer.args.report_to=none \
+                                        trainer.method_args.noise_std=${r2d_noise_std} \
+                                        trainer.method_args.noise_seed=${r2d_noise_seed} \
+                                        trainer.method_args.noise_trainable_only=${r2d_noise_trainable_only} \
+                                        trainer.method_args.dp_epsilon=${r2d_eps} \
+                                        trainer.method_args.dp_delta=${r2d_delta} \
+                                        trainer.method_args.dp_sensitivity=${r2d_sens} \
+                                        trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
+                                        trainer.method_args.r2d_L=${r2d_L} \
+                                        trainer.method_args.r2d_G=${r2d_G} \
+                                        trainer.method_args.r2d_n=${r2d_n} \
+                                        trainer.method_args.r2d_m=${r2d_m} \
+                                        trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
+                                        trainer.method_args.r2d_eta=${r2d_eta} \
+                                        retain_logs_path=null \
+                                        "${extra_train_args[@]}" \
+                                        paths.output_dir=${run_dir} \
+                                    )
+                                    if [[ "${max_steps}" != "0" ]]; then
+                                        train_cmd+=(trainer.args.max_steps=${max_steps})
+                                    fi
+
+                                    python src/train.py "${train_cmd[@]}"
+                                fi
+
+                                mkdir -p "${eval_dir}"
+                                if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
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
+                                    model.model_args.base_model_name_or_path=${rewind_model_path} \
+                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
+                                    model.model_args.device_map="auto" \
+                                    model.model_args.low_cpu_mem_usage=true \
+                                    model.lora_config.r=${lora_r} \
+                                    model.lora_config.lora_alpha=${lora_alpha} \
+                                    model.lora_config.lora_dropout=${lora_dropout} \
+                                    eval.duet.batch_size=${eval_batch_size} \
+                                    eval.duet.overwrite=true \
+                                    "${extra_eval_args[@]}" \
+                                    paths.output_dir=${eval_dir} \
+                                    retain_logs_path=null \
+                                )
+                                python src/eval.py "${eval_cmd[@]}"
+
+                                if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
+                                    if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
+                                        rm -f "${run_dir}"/*.safetensors
+                                        echo "[popqa][R2D] Removed safetensors from ${run_dir}"
+                                    fi
+                                fi
+                            done
+                        done
+                    done
                 done
             done
         done
```
