#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
repo_root=$(realpath "${script_dir}/../..")
source "${script_dir}/_splits.sh"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

parse_checkpoint_step() {
    local value="${1:-}"
    if [[ "${value}" =~ checkpoint-([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

sum_forget_m_from_split() {
    local split="$1"
    local nums
    nums=$(echo "${split}" | grep -oE 'forget[^0-9]*[0-9]+' | grep -oE '[0-9]+' || true)
    if [[ -z "${nums}" ]]; then
        echo ""
        return
    fi

    local total=0
    local n
    for n in ${nums}; do
        total=$((total + n))
    done
    echo "${total}"
}

retain_n_from_split() {
    local split="$1"
    local n
    n=$(echo "${split}" | grep -oE 'retain[^0-9]*[0-9]+' | head -n1 | grep -oE '[0-9]+' || true)
    echo "${n}"
}

base_model="${BASE_MODEL:-Llama-3.1-8B-Instruct}"
lora_model="${MODEL_CONFIG:-${base_model}-lora}"
hf_base_model_path="${HF_BASE_MODEL_PATH:-meta-llama/${base_model}}"
local_sft_base="${LOCAL_SFT_BASE:-/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb}"

sft_subfolder="${SFT_SUBFOLDER:-}"
tokenizer_subfolder="${TOKENIZER_SUBFOLDER-${sft_subfolder}}"

use_sft_base=${USE_SFT_BASE:-1}
if [[ "${use_sft_base}" == "1" ]]; then
    trained_model_path="${local_sft_base}"
    echo "[duet][R2D] Using finetuned model path ${trained_model_path}"
else
    trained_model_path="${hf_base_model_path}"
    echo "[duet][R2D] Using HF base model path ${trained_model_path}"
fi

rewind_model_path="${R2D_REWIND_CKPT_PATH:-${trained_model_path}}"
rewind_subfolder="${R2D_REWIND_SUBFOLDER:-}"

if [[ -z "${rewind_subfolder}" && -n "${R2D_REWIND_STEP:-}" ]]; then
    if [[ -n "${sft_subfolder}" ]]; then
        rewind_subfolder="${sft_subfolder}/checkpoint-${R2D_REWIND_STEP}"
    else
        rewind_subfolder="checkpoint-${R2D_REWIND_STEP}"
    fi
fi

if [[ -z "${R2D_REWIND_CKPT_PATH:-}" && -z "${R2D_REWIND_SUBFOLDER:-}" && -z "${R2D_REWIND_STEP:-}" ]]; then
    echo "[duet][R2D] ERROR: set R2D_REWIND_CKPT_PATH or R2D_REWIND_SUBFOLDER or R2D_REWIND_STEP."
    exit 1
fi

parsed_ckpt_step="$(parse_checkpoint_step "${rewind_subfolder}")"
if [[ -z "${parsed_ckpt_step}" ]]; then
    parsed_ckpt_step="$(parse_checkpoint_step "${rewind_model_path}")"
fi

if [[ -n "${R2D_REWIND_STEP:-}" && -n "${parsed_ckpt_step}" && "${R2D_REWIND_STEP}" != "${parsed_ckpt_step}" ]]; then
    echo "[duet][R2D] ERROR: R2D_REWIND_STEP=${R2D_REWIND_STEP} mismatches checkpoint step=${parsed_ckpt_step} from rewind path/subfolder."
    exit 1
fi

r2d_rewind_step_for_sigma="${R2D_REWIND_STEP_FOR_SIGMA:-}"
if [[ -z "${r2d_rewind_step_for_sigma}" && -n "${R2D_REWIND_STEP:-}" ]]; then
    r2d_rewind_step_for_sigma="${R2D_REWIND_STEP}"
fi
if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
    r2d_rewind_step_for_sigma="${parsed_ckpt_step}"
fi
if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
    r2d_rewind_step_for_sigma="null"
fi

if [[ -n "${R2D_REWIND_STEP_FOR_SIGMA:-}" && -n "${parsed_ckpt_step}" && "${R2D_REWIND_STEP_FOR_SIGMA}" != "${parsed_ckpt_step}" ]]; then
    echo "[duet][R2D] ERROR: R2D_REWIND_STEP_FOR_SIGMA=${R2D_REWIND_STEP_FOR_SIGMA} must match checkpoint step=${parsed_ckpt_step} (T-K checkpoint index)."
    exit 1
fi

echo "[duet][R2D] Rewind model path: ${rewind_model_path}"
echo "[duet][R2D] Rewind subfolder : ${rewind_subfolder}"
echo "[duet][R2D] Rewind step(T-K): ${r2d_rewind_step_for_sigma}"

tokenizer_model_path="${TOKENIZER_MODEL_PATH:-${trained_model_path}}"

extra_train_args=()
extra_eval_args=()
if [[ -n "${rewind_subfolder}" ]]; then
    extra_train_args+=(+model.model_args.subfolder=${rewind_subfolder})
    extra_eval_args+=(+model.model_args.subfolder=${rewind_subfolder})
fi
if [[ -n "${tokenizer_subfolder}" ]]; then
    extra_train_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
    extra_eval_args+=(+model.tokenizer_args.subfolder=${tokenizer_subfolder})
fi

experiment="unlearn/duet/r2d_lora.yaml"
trainer="R2D"

output_root="${repo_root}/saves/unlearn/duet/r2d"
mkdir -p "${output_root}"

set_forget_retain_splits

per_device_train_batch_size=${PER_DEVICE_TRAIN_BS:-1}
gradient_accumulation_steps=${GRAD_ACCUM:-32}
eval_batch_size=${EVAL_BATCH_SIZE:-8}
num_train_epochs=${NUM_EPOCHS:-1}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-false}

raw_lrs="${LRS:-1e-5}"
raw_lrs="${raw_lrs//,/ }"
raw_lrs="${raw_lrs//\"/}"
raw_lrs="${raw_lrs//\'/}"
read -r -a lrs <<< "${raw_lrs}"

lora_rs=(${LORA_RS:-"32"})
lora_alphas=(${LORA_ALPHAS:-"64"})
lora_dropouts=(${LORA_DROPOUTS:-"0.0"})

r2d_noise_trainable_only="${R2D_NOISE_TRAINABLE_ONLY:-true}"
r2d_delta="${R2D_DELTA:-null}"
r2d_sens="${R2D_SENS:-null}"
r2d_use_analytic_gaussian="${R2D_USE_ANALYTIC_GAUSSIAN:-true}"
r2d_L="${R2D_L:-null}"
r2d_G="${R2D_G:-null}"
r2d_n_input="${R2D_N:-null}"
r2d_m_input="${R2D_M:-null}"
r2d_eta_override="${R2D_ETA:-null}"
delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"

raw_max_steps="${R2D_MAX_STEPS_LIST:-${R2D_MAX_STEPS:-0}}"
raw_max_steps="${raw_max_steps//,/ }"
raw_max_steps="${raw_max_steps//\"/}"
raw_max_steps="${raw_max_steps//\'/}"
read -r -a max_steps_list <<< "${raw_max_steps}"

raw_noise_stds="${R2D_NOISE_STDS:-${R2D_NOISE_STD:-null}}"
raw_noise_stds="${raw_noise_stds//,/ }"
raw_noise_stds="${raw_noise_stds//\"/}"
raw_noise_stds="${raw_noise_stds//\'/}"
read -r -a noise_stds <<< "${raw_noise_stds}"

raw_epsilons="${R2D_EPSILONS:-${R2D_EPS:-null}}"
raw_epsilons="${raw_epsilons//,/ }"
raw_epsilons="${raw_epsilons//\"/}"
raw_epsilons="${raw_epsilons//\'/}"
read -r -a epsilons <<< "${raw_epsilons}"

raw_noise_seeds="${R2D_NOISE_SEEDS:-${R2D_NOISE_SEED:-0}}"
raw_noise_seeds="${raw_noise_seeds//,/ }"
raw_noise_seeds="${raw_noise_seeds//\"/}"
raw_noise_seeds="${raw_noise_seeds//\'/}"
read -r -a noise_seeds <<< "${raw_noise_seeds}"

raw_train_seeds="${TRAIN_SEEDS:-${TRAIN_SEED:-42}}"
raw_train_seeds="${raw_train_seeds//,/ }"
raw_train_seeds="${raw_train_seeds//\"/}"
raw_train_seeds="${raw_train_seeds//\'/}"
read -r -a train_seeds <<< "${raw_train_seeds}"

if [[ "${#max_steps_list[@]}" -eq 1 && "${max_steps_list[0]}" == "0" ]]; then
    echo "[duet][R2D] WARNING: R2D_MAX_STEPS=0, using NUM_EPOCHS instead of explicit K steps."
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for split in "${forget_retain_splits[@]}"; do
    read -r forget_split retain_split forget_label <<< "${split}"
    if [[ -z "${forget_label:-}" ]]; then
        forget_label="${forget_split}"
    fi

    auto_m="$(sum_forget_m_from_split "${forget_split}")"
    auto_retain_n="$(retain_n_from_split "${retain_split}")"
    auto_n=""
    if [[ -n "${auto_m}" && -n "${auto_retain_n}" ]]; then
        auto_n=$((auto_m + auto_retain_n))
    fi

    r2d_m="${r2d_m_input}"
    r2d_n="${r2d_n_input}"
    if [[ "${r2d_m}" == "null" && -n "${auto_m}" ]]; then
        r2d_m="${auto_m}"
    fi
    if [[ "${r2d_n}" == "null" && -n "${auto_n}" ]]; then
        r2d_n="${auto_n}"
    fi

    for max_steps in "${max_steps_list[@]}"; do
        if [[ "${max_steps}" == "0" ]]; then
            k_tag="ep${num_train_epochs}"
        else
            k_tag="k${max_steps}"
        fi

        for r2d_noise_std in "${noise_stds[@]}"; do
            eps_candidates=("null")
            if [[ "${r2d_noise_std}" == "null" ]]; then
                eps_candidates=("${epsilons[@]}")
            fi

            for r2d_eps in "${eps_candidates[@]}"; do
                if [[ "${r2d_noise_std}" != "null" ]]; then
                    sigma_tag="s${r2d_noise_std//./p}"
                    noise_mode="direct"
                else
                    if [[ "${r2d_eps}" == "null" ]]; then
                        echo "[duet][R2D] ERROR: DP-mode requires non-null epsilon (set R2D_EPSILONS)."
                        exit 1
                    fi
                    if [[ "${r2d_delta}" == "null" ]]; then
                        echo "[duet][R2D] ERROR: DP-mode requires non-null delta (R2D_DELTA)."
                        exit 1
                    fi
                    if [[ "${r2d_sens}" == "null" ]]; then
                        if [[ "${r2d_rewind_step_for_sigma}" == "null" ]]; then
                            echo "[duet][R2D] ERROR: DP-mode requires rewind checkpoint step (T-K) when R2D_SENS is null."
                            exit 1
                        fi
                        if [[ "${r2d_L}" == "null" || "${r2d_G}" == "null" || "${r2d_n}" == "null" || "${r2d_m}" == "null" ]]; then
                            echo "[duet][R2D] ERROR: DP-mode needs either R2D_SENS or full paper inputs (R2D_L,R2D_G,R2D_N,R2D_M,rewind_step)."
                            exit 1
                        fi
                    fi

                    eps_tag=${r2d_eps//./p}
                    delta_tag=${r2d_delta//./p}
                    sigma_tag="dp_e${eps_tag}_d${delta_tag}"
                    if [[ "${r2d_sens}" != "null" ]]; then
                        sigma_tag="${sigma_tag}_gs${r2d_sens//./p}"
                    else
                        sigma_tag="${sigma_tag}_rw${r2d_rewind_step_for_sigma}"
                    fi
                    noise_mode="dp"
                fi

                for lr in "${lrs[@]}"; do
                    r2d_eta="${r2d_eta_override}"
                    if [[ "${r2d_eta}" == "null" ]]; then
                        r2d_eta="${lr}"
                    fi

                    for lora_r in "${lora_rs[@]}"; do
                        for lora_alpha in "${lora_alphas[@]}"; do
                            for lora_dropout in "${lora_dropouts[@]}"; do
                                dropout_tag=${lora_dropout//./p}
                                rewind_tag="${R2D_REWIND_TAG:-rewind}"

                                for train_seed in "${train_seeds[@]}"; do
                                    for r2d_noise_seed in "${noise_seeds[@]}"; do
                                        seed_tag="ns${r2d_noise_seed}_ts${train_seed}"
                                        task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_${k_tag}_lr${lr}_sigma${sigma_tag}_${seed_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
                                        run_dir=${output_root}/${task_name}
                                        eval_dir=${run_dir}/evals
                                        summary_path=${eval_dir}/DUET_SUMMARY.json

                                        if [[ -f "${summary_path}" && "${FORCE_RERUN:-0}" != "1" ]]; then
                                            echo "[duet][R2D] Skipping ${task_name}: found existing summary at ${summary_path}"
                                            continue
                                        fi

                                        echo "[duet][R2D] ${task_name}"
                                        echo "  rewind=${rewind_model_path} subfolder=${rewind_subfolder} rewind_step_for_sigma=${r2d_rewind_step_for_sigma}"
                                        echo "  forget=${forget_split} retain=${retain_split} epochs=${num_train_epochs} max_steps=${max_steps} mode=${noise_mode} eps=${r2d_eps} n=${r2d_n} m=${r2d_m}"

                                        adapter_path=${run_dir}/adapter_model.safetensors
                                        if [[ ! -f "${adapter_path}" || "${FORCE_RERUN:-0}" == "1" ]]; then
                                            mkdir -p "${run_dir}"

                                            train_cmd=( \
                                                --config-name=unlearn.yaml \
                                                experiment=${experiment} \
                                                trainer=${trainer} \
                                                task_name=${task_name} \
                                                model=${lora_model} \
                                                forget_split=${forget_split} \
                                                retain_split=${retain_split} \
                                                holdout_split=${retain_split} \
                                                model.model_args.pretrained_model_name_or_path=${rewind_model_path} \
                                                model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                                model.model_args.device_map="auto" \
                                                model.model_args.low_cpu_mem_usage=true \
                                                model.lora_config.r=${lora_r} \
                                                model.lora_config.lora_alpha=${lora_alpha} \
                                                model.lora_config.lora_dropout=${lora_dropout} \
                                                trainer.args.seed=${train_seed} \
                                                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                                                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                                                trainer.args.num_train_epochs=${num_train_epochs} \
                                                trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                                                trainer.args.learning_rate=${lr} \
                                                trainer.args.optim=sgd \
                                                trainer.args.weight_decay=0.0 \
                                                trainer.args.lr_scheduler_type=constant \
                                                trainer.args.warmup_ratio=0.0 \
                                                trainer.args.save_strategy=no \
                                                trainer.args.eval_strategy=no \
                                                trainer.args.do_eval=false \
                                                trainer.args.eval_on_start=false \
                                                trainer.args.report_to=none \
                                                trainer.method_args.noise_std=${r2d_noise_std} \
                                                trainer.method_args.noise_seed=${r2d_noise_seed} \
                                                trainer.method_args.noise_trainable_only=${r2d_noise_trainable_only} \
                                                trainer.method_args.dp_epsilon=${r2d_eps} \
                                                trainer.method_args.dp_delta=${r2d_delta} \
                                                trainer.method_args.dp_sensitivity=${r2d_sens} \
                                                trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
                                                trainer.method_args.r2d_L=${r2d_L} \
                                                trainer.method_args.r2d_G=${r2d_G} \
                                                trainer.method_args.r2d_n=${r2d_n} \
                                                trainer.method_args.r2d_m=${r2d_m} \
                                                trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
                                                trainer.method_args.r2d_eta=${r2d_eta} \
                                                retain_logs_path=null \
                                                "${extra_train_args[@]}" \
                                                paths.output_dir=${run_dir} \
                                            )
                                            if [[ "${max_steps}" != "0" ]]; then
                                                train_cmd+=(trainer.args.max_steps=${max_steps})
                                            fi

                                            python src/train.py "${train_cmd[@]}"
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
                                            model.model_args.base_model_name_or_path=${rewind_model_path} \
                                            model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_model_path} \
                                            model.model_args.device_map="auto" \
                                            model.model_args.low_cpu_mem_usage=true \
                                            model.lora_config.r=${lora_r} \
                                            model.lora_config.lora_alpha=${lora_alpha} \
                                            model.lora_config.lora_dropout=${lora_dropout} \
                                            eval.duet.batch_size=${eval_batch_size} \
                                            eval.duet.overwrite=true \
                                            "${extra_eval_args[@]}" \
                                            paths.output_dir=${eval_dir} \
                                            retain_logs_path=null \
                                        )
                                        python src/eval.py "${eval_cmd[@]}"

                                        if [[ "${delete_model_safetensors_after_eval}" == "1" ]]; then
                                            if compgen -G "${run_dir}/*.safetensors" > /dev/null; then
                                                rm -f "${run_dir}"/*.safetensors
                                                echo "[duet][R2D] Removed safetensors from ${run_dir}"
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
done
