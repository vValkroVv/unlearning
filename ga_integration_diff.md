# GA Integration Diff

## 2026-03-12 Current Production Baseline Update

Updated files:
- `configs/experiment/unlearn/duet/grad_ascent_lora.yaml`
- `configs/experiment/unlearn/rwku/grad_ascent_lora.yaml`
- `scripts/duet/ga_duet.sh`
- `scripts/rwku/ga_rwku.sh`
- `prod-gpu-runs-new.md`

What changed:
- `gradient_checkpointing` default is now `false` in the active GA experiment configs.
- `NUM_EPOCHS` default is now `2` for current DUET/RWKU GA scripts.
- `LRS` default sweep is now `1e-6 5e-6 1e-5 5e-5 1e-4`.
- RWKU GA moved to `Llama-3.1-8B-Instruct` for the current production run stack.
- Current production LoRA defaults now use only attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 2026-03-12 Qwen/Gemma LoRA Alignment

Updated files:
- `configs/model/Qwen2.5-7B-Instruct-lora.yaml`
- `configs/model/gemma-7b-it-lora.yaml`

What changed:
- Qwen2.5-7B-Instruct and gemma-7b-it LoRA configs were aligned with the current production attention-only adapter policy.
- Default target modules are now `q_proj`, `k_proj`, `v_proj`, `o_proj`.
