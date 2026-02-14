pop_static_wga
================

Overview
- This method runs the existing WGA unlearning trainer but sets a static retain coefficient (alpha) based on the popularity of the forget split:
  - Popular split (city_forget_popular_5): alpha = 0.5
  - Rare split (city_forget_rare_5): alpha = 1.0

What it does
- Uses the preset experiment config at `configs/experiment/unlearn/duet/wga_lora.yaml`.
- Overrides `trainer.method_args.alpha` dynamically per split at launch time.
- Saves training and evaluation outputs under `saves/unlearn/pop_static_wga/<task_name>`.

How to run
- Edit `run.sh` to point to your desired base model path(s).
- Run the script:
  - `bash community/methods/pop_static_wga/run.sh`

Notes
- The underlying trainer is unchanged (still WGA); only alpha varies based on the forget split.
- You can customize learning rate, LoRA ranks, and other hyperparameters in the script.
