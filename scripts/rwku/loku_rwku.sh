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

echo "[rwku][LoKU] Using Hugging Face base checkpoint ${base_model_path}"

experiment="unlearn/rwku/loku_lora.yaml"
trainer="LoKU"

output_root="${OUTPUT_ROOT:-${repo_root}/saves/unlearn/rwku/loku}"
if [[ -n "${OUTPUT_ROOT:-}" && -z "${IMPORTANCE_ROOT:-}" ]]; then
    importance_root="$(dirname "${OUTPUT_ROOT}")/importances/rwku/loku"
else
    importance_root="${IMPORTANCE_ROOT:-${repo_root}/saves/importances/rwku/loku}"
fi
importance_path_template="${IMPORTANCE_PATH:-}"
fila_base_root="${FILA_BASE_ROOT:-}"
fila_base_path_template="${FILA_BASE_PATH:-}"
mkdir -p "${output_root}" "${importance_root}"
if [[ -n "${fila_base_root}" ]]; then
    mkdir -p "${fila_base_root}"
fi

forget_split="${FORGET_SPLIT:-forget_level2}"
retain_split="${RETAIN_SPLIT:-neighbor_level2}"
forget_label="${FORGET_LABEL:-${forget_split}}"

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
importance_batch_size=${IMPORTANCE_BATCH_SIZE:-1}
importance_max_steps=${IMPORTANCE_MAX_STEPS:-0}
eval_batch_size=${EVAL_BATCH_SIZE:-8}
num_train_epochs=${NUM_EPOCHS:-2}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}
max_steps="${MAX_STEPS:-0}"
checkpoint_every_half_epoch="${CHECKPOINT_EVERY_HALF_EPOCH:-1}"
save_total_limit="${SAVE_TOTAL_LIMIT:-12}"

raw_lrs="${LRS:-1e-6 5e-6 1e-5 5e-5 1e-4}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_ihl_alphas="${IHL_ALPHAS:-1.0}"
raw_ihl_alphas="${raw_ihl_alphas//,/ }"
raw_ihl_alphas="${raw_ihl_alphas//\"/}"
raw_ihl_alphas="${raw_ihl_alphas//\'/}"
read -r -a ihl_alphas <<< "${raw_ihl_alphas}"

raw_alphas="${ALPHAS:-1.0}"
raw_alphas="${raw_alphas//,/ }"
raw_alphas="${raw_alphas//\"/}"
raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1.0}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

fila_eps="${FILA_EPS:-1e-5}"
fila_adapter_name="${FILA_ADAPTER_NAME:-default}"
fila_base_subdir="${FILA_BASE_SUBDIR:-base_model}"
run_fila_sanity_check="${RUN_FILA_SANITY_CHECK:-true}"

loku_target_modules="${LOKU_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
loku_weight_decay="${LOKU_WEIGHT_DECAY:-0.01}"
loku_lr_scheduler_type="${LOKU_LR_SCHEDULER_TYPE:-linear}"
loku_warmup_epochs="${LOKU_WARMUP_EPOCHS:-1.0}"
loku_warmup_ratio="${LOKU_WARMUP_RATIO:-0.0}"

targets_tag="${LOKU_TARGETS_TAG:-no_lm_head_lora_targets}"
force_importance="${FORCE_IMPORTANCE_RECOMPUTE:-0}"
force_rerun="${FORCE_RERUN:-0}"

raw_lora_rs="${LORA_RS:-32}"
raw_lora_rs="${raw_lora_rs//,/ }"
raw_lora_rs="${raw_lora_rs//\"/}"
raw_lora_rs="${raw_lora_rs//\'/}"
read -r -a lora_rs <<< "${raw_lora_rs}"

raw_lora_alphas="${LORA_ALPHAS:-64}"
raw_lora_alphas="${raw_lora_alphas//,/ }"
raw_lora_alphas="${raw_lora_alphas//\"/}"
raw_lora_alphas="${raw_lora_alphas//\'/}"
read -r -a lora_alphas <<< "${raw_lora_alphas}"

raw_lora_dropouts="${LORA_DROPOUTS:-0.0}"
raw_lora_dropouts="${raw_lora_dropouts//,/ }"
raw_lora_dropouts="${raw_lora_dropouts//\"/}"
raw_lora_dropouts="${raw_lora_dropouts//\'/}"
read -r -a lora_dropouts <<< "${raw_lora_dropouts}"

# Keep importance LoRA config aligned with training defaults unless explicitly overridden.
importance_lora_r="${IMPORTANCE_LORA_R:-${lora_rs[0]}}"
importance_lora_alpha="${IMPORTANCE_LORA_ALPHA:-${lora_alphas[0]}}"
importance_lora_dropout="${IMPORTANCE_LORA_DROPOUT:-${lora_dropouts[0]}}"
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
delete_importance_after_run="${DELETE_IMPORTANCE_AFTER_RUN:-0}"
delete_fila_base_after_eval="${DELETE_FILA_BASE_AFTER_EVAL:-0}"

if [[ "${checkpoint_every_half_epoch}" == "1" && "${delete_fila_base_after_eval}" == "1" ]]; then
    echo "[rwku][LoKU] CHECKPOINT_EVERY_HALF_EPOCH=1 requires keeping FILA base_model for later checkpoint eval; overriding DELETE_FILA_BASE_AFTER_EVAL=0"
    delete_fila_base_after_eval=0
fi

importance_cleanup_paths=()
fila_base_cleanup_paths=()

resolve_importance_path() {
    local forget_label="$1"
    local retain_split="$2"
    local path="${importance_root}/${base_model}_${forget_label}_${retain_split}_${targets_tag}.pt"
    if [[ -n "${importance_path_template}" ]]; then
        path="${importance_path_template}"
        path="${path//\{base_model\}/${base_model}}"
        path="${path//\{forget_label\}/${forget_label}}"
        path="${path//\{retain_split\}/${retain_split}}"
        path="${path//\{targets_tag\}/${targets_tag}}"
    fi
    echo "${path}"
}

resolve_fila_base_path() {
    local forget_label="$1"
    local retain_split="$2"
    local task_name="$3"
    local run_dir="$4"
    local path

    if [[ -n "${fila_base_path_template}" ]]; then
        path="${fila_base_path_template}"
        path="${path//\{base_model\}/${base_model}}"
        path="${path//\{forget_label\}/${forget_label}}"
        path="${path//\{retain_split\}/${retain_split}}"
        path="${path//\{task_name\}/${task_name}}"
    elif [[ -n "${fila_base_root}" ]]; then
        path="${fila_base_root}/${task_name}"
    elif [[ "${fila_base_subdir}" = /* ]]; then
        path="${fila_base_subdir}"
    else
        path="${run_dir}/${fila_base_subdir}"
    fi

    echo "${path}"
}

register_importance_cleanup_path() {
    local path="$1"
    local existing
    for existing in "${importance_cleanup_paths[@]}"; do
        if [[ "${existing}" == "${path}" ]]; then
            return
        fi
    done
    importance_cleanup_paths+=("${path}")
}

register_fila_base_cleanup_path() {
    local path="$1"
    local existing
    for existing in "${fila_base_cleanup_paths[@]}"; do
        if [[ "${existing}" == "${path}" ]]; then
            return
        fi
    done
    fila_base_cleanup_paths+=("${path}")
}

cleanup_importance_files() {
    if [[ "${delete_importance_after_run}" != "1" ]]; then
        return
    fi
    local path
    for path in "${importance_cleanup_paths[@]}"; do
        if [[ -f "${path}" ]]; then
            rm -f "${path}"
            echo "[rwku][LoKU] Removed importance file ${path}"
        fi
    done
}

remove_fila_base_dir() {
    local path="$1"
    if [[ -z "${path}" || "${path}" == "/" ]]; then
        echo "[rwku][LoKU] Refusing to remove unsafe FILA base path '${path}'"
        return
    fi
    if [[ -d "${path}" ]]; then
        rm -rf "${path}"
        echo "[rwku][LoKU] Removed FILA base model dir ${path}"
    fi
}

cleanup_fila_base_dirs() {
    if [[ "${delete_fila_base_after_eval}" != "1" ]]; then
        return
    fi
    local path
    for path in "${fila_base_cleanup_paths[@]}"; do
        remove_fila_base_dir "${path}"
    done
}

cleanup_loku_artifacts() {
    cleanup_importance_files
    cleanup_fila_base_dirs
}

trap cleanup_loku_artifacts EXIT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

imp_path=$(resolve_importance_path "${forget_label}" "${retain_split}")
mkdir -p "$(dirname "${imp_path}")"
register_importance_cleanup_path "${imp_path}"
if [[ ! -f "${imp_path}" || "${force_importance}" == "1" ]]; then
    echo "[rwku][LoKU] Measuring importance -> ${imp_path}"
    python src/tools/loku_measure_importance.py \
        --config-name unlearn.yaml \
        --experiment=${experiment} \
        --output-path="${imp_path}" \
        --max-steps=${importance_max_steps} \
        --batch-size=${importance_batch_size} \
        --seed=${TRAIN_SEED:-42} \
        -- \
        model=${lora_model} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        model.model_args.pretrained_model_name_or_path=${base_model_path} \
        model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
        model.model_args.device_map=null \
        ++model.model_args.low_cpu_mem_usage=true \
        "model.lora_config.target_modules=${loku_target_modules}" \
        model.lora_config.r=${importance_lora_r} \
        model.lora_config.lora_alpha=${importance_lora_alpha} \
        model.lora_config.lora_dropout=${importance_lora_dropout} \
        trainer.args.per_device_train_batch_size=${importance_batch_size} \
        trainer.args.gradient_accumulation_steps=1 \
        trainer.args.gradient_checkpointing=false \
        trainer.args.num_train_epochs=1 \
        retain_logs_path=null
fi

for lr in "${lrs[@]}"; do
    for ihl_alpha in "${ihl_alphas[@]}"; do
        ihl_tag=${ihl_alpha//./p}
        for alpha in "${alphas[@]}"; do
            alpha_tag=${alpha//./p}
            for gamma in "${gammas[@]}"; do
                gamma_tag=${gamma//./p}
                for lora_r in "${lora_rs[@]}"; do
                    for lora_alpha in "${lora_alphas[@]}"; do
                        for lora_dropout in "${lora_dropouts[@]}"; do
                            dropout_tag=${lora_dropout//./p}
                            task_name=rwku_${base_model}_${forget_label}_loku_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_ihla${ihl_tag}_alpha${alpha_tag}_gamma${gamma_tag}
                            run_dir=${output_root}/${task_name}
                            eval_dir=${run_dir}/evals
                            summary_path=${eval_dir}/DUET_SUMMARY.json

                            if [[ -f "${summary_path}" && "${force_rerun}" != "1" ]]; then
                                echo "[rwku][LoKU] Skipping ${task_name}: found existing summary at ${summary_path}"
                                continue
                            fi

                            base_residual_dir=$(resolve_fila_base_path "${forget_label}" "${retain_split}" "${task_name}" "${run_dir}")
                            register_fila_base_cleanup_path "${base_residual_dir}"

                            echo "[rwku][LoKU] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                            adapter_path=${run_dir}/adapter_model.safetensors
                            if [[ ! -f "${adapter_path}" || ! -d "${base_residual_dir}" || "${force_rerun}" == "1" ]]; then
                                mkdir -p "${run_dir}"
                                mkdir -p "$(dirname "${base_residual_dir}")"
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
                                        trainer.args.save_strategy=steps
                                        trainer.args.save_steps=${half_epoch_steps}
                                        trainer.args.save_total_limit=${save_total_limit}
                                        trainer.args.logging_strategy=steps
                                        trainer.args.logging_steps=${logging_steps}
                                        trainer.args.save_safetensors=true
                                        trainer.args.load_best_model_at_end=false
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
                                    "model.lora_config.target_modules=${loku_target_modules}"
                                    model.lora_config.r=${lora_r}
                                    model.lora_config.lora_alpha=${lora_alpha}
                                    model.lora_config.lora_dropout=${lora_dropout}
                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size}
                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}
                                    trainer.args.num_train_epochs=${num_train_epochs}
                                    trainer.args.gradient_checkpointing=${gradient_checkpointing}
                                    trainer.args.learning_rate=${lr}
                                    trainer.args.weight_decay=${loku_weight_decay}
                                    trainer.args.lr_scheduler_type=${loku_lr_scheduler_type}
                                    +trainer.args.warmup_epochs=${loku_warmup_epochs}
                                    trainer.args.warmup_ratio=${loku_warmup_ratio}
                                    trainer.method_args.ihl_alpha=${ihl_alpha}
                                    trainer.method_args.alpha=${alpha}
                                    trainer.method_args.gamma=${gamma}
                                    trainer.method_args.retain_loss_type=NLL
                                    trainer.method_args.importance_file=${imp_path}
                                    trainer.method_args.fila_eps=${fila_eps}
                                    trainer.method_args.fila_adapter_name=${fila_adapter_name}
                                    trainer.method_args.fila_base_subdir=${base_residual_dir}
                                    trainer.method_args.run_fila_sanity_check=${run_fila_sanity_check}
                                    trainer.trace_jsonl=true
                                    retain_logs_path=null
                                    "${extra_schedule_args[@]}"
                                    paths.output_dir=${run_dir}
                                )
                                if [[ "${max_steps}" != "0" ]]; then
                                    train_cmd+=(+trainer.args.max_steps=${max_steps})
                                fi
                                python "${train_cmd[@]}"
                            fi

                            mkdir -p "${eval_dir}"
                            if [[ "${force_rerun}" == "1" ]]; then
                                rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                            fi

                            eval_cmd=( \
                                experiment=eval/rwku/default.yaml \
                                model=${lora_model} \
                                forget_split=${forget_split} \
                                holdout_split=${retain_split} \
                                task_name=${task_name} \
                                model.model_args.pretrained_model_name_or_path=${run_dir} \
                                ++model.model_args.base_model_name_or_path=${base_residual_dir} \
                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                model.model_args.device_map="auto" \
                                ++model.model_args.low_cpu_mem_usage=true \
                                "model.lora_config.target_modules=${loku_target_modules}" \
                                model.lora_config.r=${lora_r} \
                                model.lora_config.lora_alpha=${lora_alpha} \
                                model.lora_config.lora_dropout=${lora_dropout} \
                                eval.duet.batch_size=${eval_batch_size} \
                                eval.duet.overwrite=true \
                                paths.output_dir=${eval_dir} \
                                retain_logs_path=null \
                            )
                            python src/eval.py "${eval_cmd[@]}"

                            if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                    rm -f "${run_dir}"/*.safetensors
                                    echo "[rwku][LoKU] Removed safetensors from ${run_dir}"
                                fi
                            fi

                            if [[ "${delete_fila_base_after_eval}" == "1" ]]; then
                                remove_fila_base_dir "${base_residual_dir}"
                            fi
                        done
                    done
                done
            done
        done
    done
done
