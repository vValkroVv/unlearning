#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${hf_base_model_path}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    base_model_path="${local_sft_base}"
    echo "[duet][FALCON] Using locally finetuned base checkpoint at ${base_model_path}"
else
    base_model_path="${hf_base_model_path}"
    echo "[duet][FALCON] Using Hugging Face base checkpoint ${base_model_path}"
fi

experiment="unlearn/duet/falcon_lora.yaml"
trainer="FALCON"

output_root="${repo_root}/saves/unlearn/duet/falcon"
mkdir -p "${output_root}"

# Match NPO/GA run style: merge rare+popular forget splits by default.
export MERGE_POPULARITY_FORGET=${MERGE_POPULARITY_FORGET:-1}
set_forget_retain_splits

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
num_train_epochs=${NUM_EPOCHS:-5}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}

raw_lrs="${LRS:-1e-5 5e-5 1e-4 5e-4 1e-3}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

raw_temps="${TEMPS:-0.07}"
raw_temps="${raw_temps//,/ }"
raw_temps="${raw_temps//\"/}"
raw_temps="${raw_temps//\'/}"
read -r -a temps <<< "${raw_temps}"

raw_k_svds="${K_SVDS:-2,4,8,16}"
raw_k_svds="${raw_k_svds//,/ }"
raw_k_svds="${raw_k_svds//\"/}"
raw_k_svds="${raw_k_svds//\'/}"
read -r -a k_svds <<< "${raw_k_svds}"

raw_pov_alphas="${POV_ALPHAS:-1.0}"
raw_pov_alphas="${raw_pov_alphas//,/ }"
raw_pov_alphas="${raw_pov_alphas//\"/}"
raw_pov_alphas="${raw_pov_alphas//\'/}"
read -r -a pov_alphas <<< "${raw_pov_alphas}"

raw_pov_noise_stds="${POV_NOISE_STDS:-0.0}"
raw_pov_noise_stds="${raw_pov_noise_stds//,/ }"
raw_pov_noise_stds="${raw_pov_noise_stds//\"/}"
raw_pov_noise_stds="${raw_pov_noise_stds//\'/}"
read -r -a pov_noise_stds <<< "${raw_pov_noise_stds}"

raw_pov_transforms="${POV_TRANSFORMS:-tanh}"
raw_pov_transforms="${raw_pov_transforms//,/ }"
raw_pov_transforms="${raw_pov_transforms//\"/}"
raw_pov_transforms="${raw_pov_transforms//\'/}"
read -r -a pov_transforms <<< "${raw_pov_transforms}"

raw_target_layers="${TARGET_LAYERS:-7}"
raw_target_layers="${raw_target_layers//,/ }"
raw_target_layers="${raw_target_layers//\"/}"
raw_target_layers="${raw_target_layers//\'/}"
read -r -a target_layers <<< "${raw_target_layers}"

raw_alphas="${ALPHAS:-1,2,4}"
raw_alphas="${raw_alphas//,/ }"
raw_alphas="${raw_alphas//\"/}"
raw_alphas="${raw_alphas//\'/}"
read -r -a alphas <<< "${raw_alphas}"

raw_gammas="${GAMMAS:-1,2,4}"
raw_gammas="${raw_gammas//,/ }"
raw_gammas="${raw_gammas//\"/}"
raw_gammas="${raw_gammas//\'/}"
read -r -a gammas <<< "${raw_gammas}"

raw_conflict_thresholds="${CONFLICT_COS_THRESHOLDS:-0.0}"
raw_conflict_thresholds="${raw_conflict_thresholds//,/ }"
raw_conflict_thresholds="${raw_conflict_thresholds//\"/}"
raw_conflict_thresholds="${raw_conflict_thresholds//\'/}"
read -r -a conflict_thresholds <<< "${raw_conflict_thresholds}"

raw_retain_modes="${RETAIN_MODES:-cosine}"
raw_retain_modes="${raw_retain_modes//,/ }"
raw_retain_modes="${raw_retain_modes//\"/}"
raw_retain_modes="${raw_retain_modes//\'/}"
read -r -a retain_modes <<< "${raw_retain_modes}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

# Optional MI-guided target-layer selection (paper-style preprocessing step).
mi_select_layers=${MI_SELECT_LAYERS:-0}
mi_model_cfg="${MI_MODEL_CFG:-${repo_root}/configs/model/Llama-3.1-8B-Instruct.yaml}"
mi_model_path="${MI_MODEL_PATH:-${base_model_path}}"
mi_tokenizer_path="${MI_TOKENIZER_PATH:-${tokenizer_model_path}}"
mi_dataset_path="${MI_DATASET_PATH:-SwetieePawsss/DUET}"
mi_question_key="${MI_QUESTION_KEY:-question}"
mi_answer_key="${MI_ANSWER_KEY:-answer}"
mi_answer_index="${MI_ANSWER_INDEX:-}"
mi_max_length="${MI_MAX_LENGTH:-512}"
mi_batch_size="${MI_BATCH_SIZE:-1}"
mi_eta="${MI_ETA:-1.0}"
mi_pca_var="${MI_PCA_VAR:-0.95}"
mi_max_examples="${MI_MAX_EXAMPLES:-200}"
raw_mi_topks="${MI_TOPK:-1,3}"
raw_mi_topks="${raw_mi_topks//,/ }"
raw_mi_topks="${raw_mi_topks//\"/}"
raw_mi_topks="${raw_mi_topks//\'/}"
read -r -a mi_topks <<< "${raw_mi_topks}"
mi_seed="${MI_SEED:-0}"
mi_device="${MI_DEVICE:-cuda}"
mi_out_dir="${MI_OUT_DIR:-${output_root}/mi_layers}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    read -r forget_split retain_split forget_label <<< "${split}"
    if [[ -z "${forget_label:-}" ]]; then
        forget_label="${forget_split}"
    fi
    mi_loop_topks=("_none")
    if [[ "${mi_select_layers}" == "1" ]]; then
        mi_loop_topks=("${mi_topks[@]}")
    fi

    for mi_topk in "${mi_loop_topks[@]}"; do
        split_target_layers=("${target_layers[@]}")
        mi_topk_tag=""

        if [[ "${mi_select_layers}" == "1" ]]; then
            mi_forget_splits_raw="${MI_FORGET_SPLITS:-${forget_split//+/ }}"
            mi_forget_splits_raw="${mi_forget_splits_raw//,/ }"
            read -r -a mi_forget_splits <<< "${mi_forget_splits_raw}"
            mi_retain_split="${MI_RETAIN_SPLIT:-${retain_split}}"
            mkdir -p "${mi_out_dir}"
            mi_out_json="${mi_out_dir}/${base_model}_${forget_label}_mi_topk${mi_topk}_layers.json"

            mi_cmd=(
                python "${repo_root}/src/tools/falcon_mi_select.py"
                --model_cfg "${mi_model_cfg}"
                --model_path "${mi_model_path}"
                --tokenizer_path "${mi_tokenizer_path}"
                --dataset_path "${mi_dataset_path}"
                --forget_splits "${mi_forget_splits[@]}"
                --retain_split "${mi_retain_split}"
                --question_key "${mi_question_key}"
                --answer_key "${mi_answer_key}"
                --max_length "${mi_max_length}"
                --batch_size "${mi_batch_size}"
                --eta "${mi_eta}"
                --pca_var "${mi_pca_var}"
                --max_examples "${mi_max_examples}"
                --topk "${mi_topk}"
                --seed "${mi_seed}"
                --device "${mi_device}"
                --out_json "${mi_out_json}"
                --print_layers
                --quiet
            )
            if [[ -n "${mi_answer_index}" ]]; then
                mi_cmd+=(--answer_index "${mi_answer_index}")
            fi

            echo "[duet][FALCON][MI] Selecting layers (topk=${mi_topk}) from splits: ${mi_forget_splits[*]} | retain: ${mi_retain_split}"
            mi_layers_str="$("${mi_cmd[@]}")"
            mi_layers_str="$(echo "${mi_layers_str}" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
            if [[ -z "${mi_layers_str}" ]]; then
                echo "[duet][FALCON][MI] Empty layer selection output; aborting."
                exit 1
            fi
            read -r -a split_target_layers <<< "${mi_layers_str}"
            mi_topk_tag="_mitk${mi_topk}"
            echo "[duet][FALCON][MI] Selected TARGET_LAYERS=${mi_layers_str}"
        fi

        for lr in "${lrs[@]}"; do
        for temp in "${temps[@]}"; do
            temp_tag=${temp//./p}
            for k_svd in "${k_svds[@]}"; do
                for pov_alpha in "${pov_alphas[@]}"; do
                    pov_alpha_tag=${pov_alpha//./p}
                    for pov_noise_std in "${pov_noise_stds[@]}"; do
                        pov_noise_tag=${pov_noise_std//./p}
                        for pov_transform in "${pov_transforms[@]}"; do
                            for target_layer in "${split_target_layers[@]}"; do
                                for alpha in "${alphas[@]}"; do
                                    alpha_tag=${alpha//./p}
                                    for gamma in "${gammas[@]}"; do
                                        gamma_tag=${gamma//./p}
                                        for conflict_thr in "${conflict_thresholds[@]}"; do
                                            conflict_tag=${conflict_thr//./p}
                                            for retain_mode in "${retain_modes[@]}"; do
                                                for lora_r in "${lora_rs[@]}"; do
                                                        for lora_alpha in "${lora_alphas[@]}"; do
                                                        for lora_dropout in "${lora_dropouts[@]}"; do
                                                            dropout_tag=${lora_dropout//./p}

                                                            task_name=duet_${base_model}_${forget_label}_falcon_lora_r${lora_r}_lalpha${lora_alpha}_ldrop${dropout_tag}_lr${lr}_t${temp_tag}_k${k_svd}_pova${pov_alpha_tag}_povn${pov_noise_tag}_pov${pov_transform}_layer${target_layer}${mi_topk_tag}_a${alpha_tag}_g${gamma_tag}_cth${conflict_tag}_rm${retain_mode}
                                                            run_dir=${output_root}/${task_name}
                                                            eval_dir=${run_dir}/evals
                                                            summary_path=${eval_dir}/DUET_SUMMARY.json

                                                            if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                                                echo "[duet][FALCON] Skipping ${task_name}: found existing summary at ${summary_path}"
                                                                continue
                                                            fi

                                                            echo "[duet][FALCON] ${task_name}: unlearning ${base_model_path} on ${forget_split}"

                                                            adapter_path=${run_dir}/adapter_model.safetensors
                                                            if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                                                mkdir -p "${run_dir}"
                                                                python src/train.py --config-name=unlearn.yaml \
                                                                    experiment=${experiment} \
                                                                    trainer=${trainer} \
                                                                    task_name=${task_name} \
                                                                    model=${lora_model} \
                                                                    forget_split=${forget_split} \
                                                                    retain_split=${retain_split} \
                                                                    model.model_args.pretrained_model_name_or_path=${base_model_path} \
                                                                    model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                                                    model.model_args.device_map="auto" \
                                                                    model.model_args.low_cpu_mem_usage=true \
                                                                    model.lora_config.r=${lora_r} \
                                                                    model.lora_config.lora_alpha=${lora_alpha} \
                                                                    model.lora_config.lora_dropout=${lora_dropout} \
                                                                    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                                                    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                                                    trainer.args.num_train_epochs=${num_train_epochs} \
                                                                    trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                                                    trainer.args.learning_rate=${lr} \
                                                                    trainer.method_args.temperature=${temp} \
                                                                    trainer.method_args.k_svd=${k_svd} \
                                                                    trainer.method_args.pov_alpha=${pov_alpha} \
                                                                    trainer.method_args.pov_noise_std=${pov_noise_std} \
                                                                    trainer.method_args.pov_transform=${pov_transform} \
                                                                    trainer.method_args.target_layer=${target_layer} \
                                                                    trainer.method_args.alpha=${alpha} \
                                                                    trainer.method_args.gamma=${gamma} \
                                                                    trainer.method_args.conflict_cos_threshold=${conflict_thr} \
                                                                    trainer.method_args.retain_mode=${retain_mode} \
                                                                    retain_logs_path=null \
                                                                    paths.output_dir=${run_dir}
                                                            fi

                                                            mkdir -p "${eval_dir}"
                                                            if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
                                                                rm -f "${summary_path}" "${eval_dir}/DUET_EVAL.json"
                                                            fi

                                                            eval_cmd=( \
                                                                experiment=eval/duet/default.yaml \
                                                                model=${lora_model} \
                                                                forget_split=${forget_split} \
                                                                holdout_split=${retain_split} \
                                                                task_name=${task_name} \
                                                                model.model_args.pretrained_model_name_or_path=${run_dir} \
                                                                model.model_args.base_model_name_or_path=${base_model_path} \
                                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                                                model.model_args.device_map="auto" \
                                                                model.model_args.low_cpu_mem_usage=true \
                                                                model.lora_config.r=${lora_r} \
                                                                model.lora_config.lora_alpha=${lora_alpha} \
                                                                model.lora_config.lora_dropout=${lora_dropout} \
                                                                eval.duet.overwrite=true \
                                                                paths.output_dir=${eval_dir} \
                                                                retain_logs_path=null \
                                                            )
                                                            python src/eval.py "${eval_cmd[@]}"
                done
            done
        done
    done
done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
