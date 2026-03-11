# NPO Integration Diff

## 2026-03-12 Current Production Baseline Update

Updated files:
- `configs/experiment/unlearn/duet/grad_ascent_lora.yaml`
- `configs/experiment/unlearn/rwku/npo_lora.yaml`
- `scripts/duet/npo_duet.sh`
- `scripts/rwku/npo_rwku.sh`
- `prod-gpu-runs-new.md`

What changed:
- Active NPO runs now default to `NUM_EPOCHS=2`.
- Active NPO scripts now default to `LRS="1e-6 5e-6 1e-5 5e-5 1e-4"`.
- DUET NPO keeps reusing the shared DUET grad-ascent experiment config, which now defaults `gradient_checkpointing` to `false`.
- RWKU NPO config now defaults `gradient_checkpointing` to `false` and uses `Llama-3.1-8B-Instruct`.
- Current production LoRA defaults now use only attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 2026-03-12 Qwen/Gemma LoRA Alignment

Updated files:
- `configs/model/Qwen2.5-7B-Instruct-lora.yaml`
- `configs/model/gemma-7b-it-lora.yaml`

What changed:
- Qwen2.5-7B-Instruct and gemma-7b-it LoRA configs were aligned with the active attention-only adapter policy.
- Default target modules are now `q_proj`, `k_proj`, `v_proj`, `o_proj`.
