#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")

resolve_num_rows() {
    local dataset_name="$1"
    python - <<PY
import datasets
dataset_name = ${dataset_name@Q}
ds = datasets.load_dataset("SwetieePawsss/exp_r", name=dataset_name, split="test")
print(len(ds))
PY
}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${base_model_path}}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "[rwku][STAT] Using Hugging Face base checkpoint ${base_model_path}"

experiment="unlearn/rwku/stat_lora.yaml"
trainer="STAT"

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/rwku/stat}"
mkdir -p "${output_root}"

forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-32}
gradient_accumulation_steps=${GRAD_ACCUM:-1}
eval_batch_size=${EVAL_BATCH_SIZE:-192}
num_train_epochs=${NUM_EPOCHS:-2}
max_steps="${MAX_STEPS:-0}"
checkpoint_every_half_epoch="${CHECKPOINT_EVERY_HALF_EPOCH:-1}"
save_total_limit="${SAVE_TOTAL_LIMIT:-12}"
checkpoint_epochs_raw="${CHECKPOINT_EPOCHS:-}"
checkpoint_epochs_csv=""
if [[ -n "${checkpoint_epochs_raw}" ]]; then
    checkpoint_epochs_raw="${checkpoint_epochs_raw//,/ }"
    checkpoint_epochs_raw="${checkpoint_epochs_raw//\"/}"
    checkpoint_epochs_raw="${checkpoint_epochs_raw//\'/}"
    read -r -a checkpoint_epochs <<< "${checkpoint_epochs_raw}"
    checkpoint_epochs_csv=$(IFS=,; echo "${checkpoint_epochs[*]}")
fi
run_tag_extra="${RUN_TAG_EXTRA:-}"

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_stat_forget_weights="${STAT_FORGET_WEIGHTS:-1.0}"
raw_stat_forget_weights="${raw_stat_forget_weights//,/ }"
raw_stat_forget_weights="${raw_stat_forget_weights//\"/}"
raw_stat_forget_weights="${raw_stat_forget_weights//\'/}"
read -r -a stat_forget_weights <<< "${raw_stat_forget_weights}"

raw_stat_retain_weights="${STAT_RETAIN_WEIGHTS:-1.0}"
raw_stat_retain_weights="${raw_stat_retain_weights//,/ }"
raw_stat_retain_weights="${raw_stat_retain_weights//\"/}"
raw_stat_retain_weights="${raw_stat_retain_weights//\'/}"
read -r -a stat_retain_weights <<< "${raw_stat_retain_weights}"

raw_stat_modes="${STAT_SYNTHETIC_MODES:-uniform}"
raw_stat_modes="${raw_stat_modes//,/ }"
raw_stat_modes="${raw_stat_modes//\"/}"
raw_stat_modes="${raw_stat_modes//\'/}"
read -r -a stat_modes <<< "${raw_stat_modes}"

raw_exclude_special="${STAT_EXCLUDE_SPECIAL_TOKENS:-true}"
raw_exclude_special="${raw_exclude_special//,/ }"
raw_exclude_special="${raw_exclude_special//\"/}"
raw_exclude_special="${raw_exclude_special//\'/}"
read -r -a stat_exclude_special_values <<< "${raw_exclude_special}"

raw_preserve_eos="${STAT_PRESERVE_EOS:-false}"
raw_preserve_eos="${raw_preserve_eos//,/ }"
raw_preserve_eos="${raw_preserve_eos//\"/}"
raw_preserve_eos="${raw_preserve_eos//\'/}"
read -r -a stat_preserve_eos_values <<< "${raw_preserve_eos}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for lr in "${lrs[@]}"; do
    for stat_forget_weight in "${stat_forget_weights[@]}"; do
        sfw_tag=${stat_forget_weight//./p}
        for stat_retain_weight in "${stat_retain_weights[@]}"; do
            srw_tag=${stat_retain_weight//./p}
            for stat_mode in "${stat_modes[@]}"; do
                for stat_exclude_special in "${stat_exclude_special_values[@]}"; do
                    xsp_tag="${stat_exclude_special}"
                    xsp_tag="${xsp_tag//true/T}"
                    xsp_tag="${xsp_tag//false/F}"
                    for stat_preserve_eos in "${stat_preserve_eos_values[@]}"; do
                        peos_tag="${stat_preserve_eos}"
                        peos_tag="${peos_tag//true/T}"
                        peos_tag="${peos_tag//false/F}"
                        for lora_r in "${lora_rs[@]}"; do
                            for lora_alpha in "${lora_alphas[@]}"; do
                                for lora_dropout in "${lora_dropouts[@]}"; do
                                    dropout_tag=${lora_dropout//./p}
                                    task_name=rwku_${base_model}_${forget_split}_stat_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_sfw${sfw_tag}_srw${srw_tag}_mode${stat_mode}_xsp${xsp_tag}_peos${peos_tag}
                                    if [[ -n "${run_tag_extra}" ]]; then
                                        task_name="${task_name}_${run_tag_extra}"
                                    fi
                                    run_dir=${output_root}/${task_name}
                                    eval_dir=${run_dir}/evals
                                    summary_path=${eval_dir}/DUET_SUMMARY.json

                                    if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                        echo "[rwku][STAT] Skipping ${task_name}: found existing summary at ${summary_path}"
                                        continue
                                    fi

                                    echo "${task_name}: STAT LoRA unlearning ${base_model_path} on ${forget_split}"

                                    adapter_path=${run_dir}/adapter_model.safetensors
                                    if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                        mkdir -p "${run_dir}"
                                        extra_schedule_args=()
                                        if [[ -n "${checkpoint_epochs_csv}" ]]; then
                                            extra_schedule_args+=(
                                                ++trainer.args.save_strategy=no
                                                ++trainer.args.save_total_limit=${save_total_limit}
                                                ++trainer.args.save_safetensors=true
                                                ++trainer.save_on_epochs=[${checkpoint_epochs_csv}]
                                            )
                                        elif [[ "${checkpoint_every_half_epoch}" == "1" && "${max_steps}" == "0" ]]; then
                                            num_rows=$(resolve_num_rows "${forget_split}")
                                            global_batch=$(( per_device_train_batch_size * gradient_accumulation_steps ))
                                            steps_per_epoch=$(( (num_rows + global_batch - 1) / global_batch ))
                                            half_epoch_steps=$(( (steps_per_epoch + 1) / 2 ))
                                            logging_steps=$(( half_epoch_steps / 2 ))
                                            if [[ "${logging_steps}" -lt 1 ]]; then
                                                logging_steps=1
                                            fi
                                            extra_schedule_args+=(
                                                ++trainer.args.save_strategy=steps
                                                ++trainer.args.save_steps=${half_epoch_steps}
                                                ++trainer.args.save_total_limit=${save_total_limit}
                                                ++trainer.args.logging_strategy=steps
                                                ++trainer.args.logging_steps=${logging_steps}
                                                ++trainer.args.save_safetensors=true
                                                ++trainer.args.load_best_model_at_end=false
                                            )
                                        fi
                                        train_cmd=(
                                            src/train.py
                                            --config-name=unlearn.yaml
                                            experiment=${experiment}
                                            trainer=${trainer}
                                            task_name=${task_name}
                                            model=${lora_model}
                                            forget_split=${forget_split}
                                            retain_split=${retain_split}
                                            model.model_args.pretrained_model_name_or_path=${base_model_path}
                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                                            model.model_args.device_map="auto"
                                            ++model.model_args.low_cpu_mem_usage=true
                                            model.lora_config.r=${lora_r}
                                            model.lora_config.lora_alpha=${lora_alpha}
                                            model.lora_config.lora_dropout=${lora_dropout}
                                            ++trainer.args.seed=${TRAIN_SEED:-42}
                                            ++trainer.args.data_seed=${DATA_SEED:-${TRAIN_SEED:-42}}
                                            trainer.args.per_device_train_batch_size=${per_device_train_batch_size}
                                            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}
                                            trainer.args.num_train_epochs=${num_train_epochs}
                                            trainer.args.learning_rate=${lr}
                                            trainer.method_args.stat_forget_weight=${stat_forget_weight}
                                            trainer.method_args.stat_retain_weight=${stat_retain_weight}
                                            trainer.method_args.synthetic_mode=${stat_mode}
                                            trainer.method_args.exclude_special_tokens=${stat_exclude_special}
                                            trainer.method_args.preserve_eos=${stat_preserve_eos}
                                            trainer.method_args.retain_loss_type=NLL
                                            +trainer.trace_jsonl=true
                                            retain_logs_path=null
                                            "${extra_schedule_args[@]}"
                                            paths.output_dir=${run_dir}
                                        )
                                        if [[ "${max_steps}" != "0" ]]; then
                                            train_cmd+=(+trainer.args.max_steps=${max_steps})
                                        fi
                                        if [[ "${FULL_DETERMINISM:-0}" == "1" ]]; then
                                            train_cmd+=(++trainer.args.full_determinism=true)
                                        fi
                                        python "${train_cmd[@]}"
                                    fi

                                    mkdir -p "${eval_dir}"
                                    if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                        rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                                    fi

                                    eval_cmd=(
                                        experiment=eval/rwku/default.yaml
                                        model=${lora_model}
                                        forget_split=${forget_split}
                                        holdout_split=${retain_split}
                                        task_name=${task_name}
                                        model.model_args.pretrained_model_name_or_path=${run_dir}
                                        ++model.model_args.base_model_name_or_path=${base_model_path}
                                        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path}
                                        model.model_args.device_map="auto"
                                        ++model.model_args.low_cpu_mem_usage=true
                                        model.lora_config.r=${lora_r}
                                        model.lora_config.lora_alpha=${lora_alpha}
                                        model.lora_config.lora_dropout=${lora_dropout}
                                        eval.duet.batch_size=${eval_batch_size}
                                        eval.duet.overwrite=true
                                        paths.output_dir=${eval_dir}
                                        retain_logs_path=null
                                    )
                                    python src/eval.py "${eval_cmd[@]}"

                                    if [[ "${run_checkpoint_eval}" == "1" ]]; then
                                        bash "${script_dir}/eval_checkpoints_rwku.sh" \
                                            "${run_dir}" \
                                            "${forget_split}" \
                                            "${retain_split}" \
                                            "${base_model_path}" \
                                            "${tokenizer_model_path}" \
                                            "${lora_model}" \
                                            "${base_model}"
                                    fi

                                    if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                        if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                            rm -f "${run_dir}"/*.safetensors
                                            echo "[rwku][STAT] Removed safetensors from ${run_dir}"
                                        fi
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
