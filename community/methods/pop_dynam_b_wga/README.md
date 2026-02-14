pop_dynam_b_wga
================

Overview
- Dynamic-WGA variant that computes beta per-sample using DUET's popularity score:
  beta_i = 2.26 * (pop_sum[i])^(-0.677)
- Requires the dataset to expose `pop_sum` per example (added to QADataset and collator).
- Uses the same retain objective as GradDiff (NLL by default, KL optional via config).

How to run
- See `run.sh` for DUET LoRA runs mirroring pop_static_wga but without manual beta.
- Outputs are saved under `saves/unlearn/pop_dynam_b_wga/<task_name>`.

Notes
- If `pop_sum` is missing, the code falls back to beta=1.0 for that batch.
- Works with `configs/experiment/unlearn/duet/wga_lora.yaml` by overriding trainer to `PopDynamBWGA`.
