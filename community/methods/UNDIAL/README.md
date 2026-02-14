# UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models (NAACL 2025)

- Authors: Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramón Huerta, Ivan Vulić
- Link​: https://arxiv.org/pdf/2402.10052

# Setup
- Hyperparameters: The original paper uses Llama-2 7B with LoRA to tune the model (rank=8, alpha=16) and learning rate of 1e-4. It's suggested to search the learning rate over [1e-5, 3e-4, 1e-4], and use an effective batch size of 32 (batch_size * gradient_accumulation). The other important hyperparemeter is beta, the strength of penalty, which typically takes a number between [3,10,30]. If we change to other models, adjusting learning rate accordingly.

- Computation Setup: All experiments are run on one A100.
- Other Details: The original paper does not use the retain set and aims to retain knowledge in all domains, not just on the retain set. So alpha is set to 0. Practionioners could search over the alpha or gamma to better retain the performance on the retain set.

# Results
Run `run.sh` for full-parameter UNDIAL, `run_lora.sh` for MUSE LoRA adapters, or `run_duet_lora.sh` for the SwetieePawsss/DUET benchmark. The LoRA scripts expose the adapter rank (`lora_rs`), alpha (`lora_alphas`), and dropout (`lora_dropouts`) arrays so you can sweep alternative configurations. They default to the public `meta-llama/Llama-3.1-8B-Instruct` checkpoint; flip the `use_sft_base` toggle near the top of the script to run against the locally finetuned model at `/mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb`. Outputs land under `saves/unlearn/<TASK_NAME>`.

# Citation
@misc{dong2024undial,
      title={UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models}, 
      author={Yijiang River Dong and Hongzhou Lin and Mikhail Belkin and Ramon Huerta and Ivan Vulić},
      year={2024},
      eprint={2402.10052},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.10052}, 
}
