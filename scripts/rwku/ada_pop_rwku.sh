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

echo "[rwku][AdaPop] Using Hugging Face base checkpoint ${base_model_path}"

experiment="${EXPERIMENT:-unlearn/rwku/ada_pop_lora.yaml}"
trainer="AdaPop"

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/rwku/ada_pop}"
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

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

raw_alpha_consts="${ALPHA_CONST:-none}"
raw_alpha_consts="${raw_alpha_consts//,/ }"
raw_alpha_consts="${raw_alpha_consts//\"/}"
raw_alpha_consts="${raw_alpha_consts//\'/}"
read -r -a alpha_consts <<< "${raw_alpha_consts}"

raw_beta_consts="${BETA_CONST:-none}"
raw_beta_consts="${raw_beta_consts//,/ }"
raw_beta_consts="${raw_beta_consts//\"/}"
raw_beta_consts="${raw_beta_consts//\'/}"
read -r -a beta_consts <<< "${raw_beta_consts}"

beta_a_override="${BETA_A:-}"
beta_b_override="${BETA_B:-}"
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
run_checkpoint_eval="${RUN_CHECKPOINT_EVAL:-${RUN_UTILITY_EVAL:-0}}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for lr in "${lrs[@]}"; do
    for gamma in "${gammas[@]}"; do
        gamma_tag=${gamma//./p}
        for alpha_const in "${alpha_consts[@]}"; do
            for beta_const in "${beta_consts[@]}"; do
                atag="adyn"
                btag="bdyn"
                beta_a_tag=""
                beta_b_tag=""
                extra_method_args=()
                shopt -s nocasematch || true
                if [[ "${alpha_const}" != "none" && -n "${alpha_const}" ]]; then
                    atag="a${alpha_const//./p}"
                    extra_method_args+=(trainer.method_args.alpha_const=${alpha_const})
                fi
                if [[ "${beta_const}" != "none" && -n "${beta_const}" ]]; then
                    btag="b${beta_const//./p}"
                    extra_method_args+=(trainer.method_args.beta_const=${beta_const})
                fi
                shopt -u nocasematch || true
                if [[ -n "${beta_a_override}" ]]; then
                    beta_a_tag="_ba${beta_a_override//./p}"
                    extra_method_args+=(trainer.method_args.beta_a=${beta_a_override})
                fi
                if [[ -n "${beta_b_override}" ]]; then
                    beta_b_tag="_bb${beta_b_override//./p}"
                    extra_method_args+=(trainer.method_args.beta_b=${beta_b_override})
                fi

                for lora_r in "${lora_rs[@]}"; do
                    for lora_alpha in "${lora_alphas[@]}"; do
                        for lora_dropout in "${lora_dropouts[@]}"; do
                            dropout_tag=${lora_dropout//./p}
                            task_name=rwku_${base_model}_${forget_split}_ada_pop_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_${atag}_${btag}${beta_a_tag}${beta_b_tag}_gamma${gamma_tag}
                            run_dir=${output_root}/${task_name}
                            eval_dir=${run_dir}/evals
                            summary_path=${eval_dir}/DUET_SUMMARY.json

                            if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                echo "[rwku][AdaPop] Skipping ${task_name}: found existing summary at ${summary_path}"
                                continue
                            fi

                            echo "${task_name}: AdaPop LoRA unlearning ${base_model_path} on ${forget_split}"

                            adapter_path=${run_dir}/adapter_model.safetensors
                            if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                mkdir -p "${run_dir}"
                                extra_schedule_args=()
                                if [[ "${checkpoint_every_half_epoch}" == "1" && "${max_steps}" == "0" ]]; then
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
                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size}
                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}
                                    trainer.args.num_train_epochs=${num_train_epochs}
                                    trainer.args.learning_rate=${lr}
                                    trainer.method_args.gamma=${gamma}
                                    trainer.method_args.retain_loss_type=NLL
                                    +trainer.trace_jsonl=true
                                    retain_logs_path=null
                                    "${extra_method_args[@]}"
                                    "${extra_schedule_args[@]}"
                                    paths.output_dir=${run_dir}
                                )
                                if [[ "${max_steps}" != "0" ]]; then
                                    train_cmd+=(+trainer.args.max_steps=${max_steps})
                                fi
                                python "${train_cmd[@]}"
                            fi

                            mkdir -p "${eval_dir}"
                            if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                            fi

                            eval_cmd=( \
                                experiment=eval/rwku/default.yaml \
                                model=${lora_model} \
                                forget_split=${forget_split} \
                                holdout_split=${retain_split} \
                                task_name=${task_name} \
                                model.model_args.pretrained_model_name_or_path=${run_dir} \
                                ++model.model_args.base_model_name_or_path=${base_model_path} \
                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                model.model_args.device_map="auto" \
                                ++model.model_args.low_cpu_mem_usage=true \
                                model.lora_config.r=${lora_r} \
                                model.lora_config.lora_alpha=${lora_alpha} \
                                model.lora_config.lora_dropout=${lora_dropout} \
                                eval.duet.batch_size=${eval_batch_size} \
                                eval.duet.overwrite=true \
                                paths.output_dir=${eval_dir} \
                                retain_logs_path=null \
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
                                    echo "[rwku][AdaPop] Removed safetensors from ${run_dir}"
                                fi
                            fi
                        done
                    done
                done
            done
        done
    done
done
