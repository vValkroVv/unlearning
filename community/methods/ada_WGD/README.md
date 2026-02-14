ada_WGD
=======

Adaptive WGA-based unlearning (AdaWGD):
- Forget: WGA with dynamic popularity beta from DUET pop_sum and optional anti-repetition.
- Retain: NLL or KL (vs a frozen reference).
- Adaptive retain constraint: alpha_k = alpha0 + lambda_k, with epoch-wise lambda updates.
- Optional warmup of forget weight gamma over Kw epochs.

Run
- Edit run.sh as needed or pass env overrides (examples below).
- Saves to: saves/unlearn/ada_WGD/<task_name>

Examples
- Default:
  bash community/methods/ada_WGD/run.sh
- Stronger forgetting and gentle retain:
  DEVICES=0 NUM_EPOCHS=10 LRS="2e-5 4e-5 6e-5" GAMMAS="1.0 3.0" \
  RETAIN_EPS=1.2 INIT_LAMBDA=0.3 DUAL_STEP_SIZE=0.05 DUAL_UPDATE_UPON=epoch DUAL_WARMUP_EPOCHS=1 \
  REP_COEFF=0.1 bash community/methods/ada_WGD/run.sh
