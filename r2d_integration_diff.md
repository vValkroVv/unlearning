# R2D Integration Diff

Base commit: 73c4007a453d8cdb39196ee9751f137dbd36e18b (before latest R2D refinements)
Target: current working tree

```diff
diff --git a/configs/experiment/unlearn/duet/r2d_lora.yaml b/configs/experiment/unlearn/duet/r2d_lora.yaml
index 63c1808..bfd09be 100644
--- a/configs/experiment/unlearn/duet/r2d_lora.yaml
+++ b/configs/experiment/unlearn/duet/r2d_lora.yaml
@@ -48,8 +48,10 @@ trainer:
     gradient_accumulation_steps: 32
     learning_rate: 1e-5
     num_train_epochs: 5
+    optim: sgd
+    weight_decay: 0.0
     lr_scheduler_type: constant
-    warmup_ratio: 0.1
+    warmup_ratio: 0.0
     logging_steps: 10
     eval_strategy: "no"
     save_strategy: "no"
@@ -59,7 +61,7 @@ trainer:
     gradient_checkpointing: false
     ddp_find_unused_parameters: false
   method_args:
-    noise_std: 0.0
+    noise_std: null
     noise_trainable_only: true
 
 task_name: duet_r2d_lora
diff --git a/configs/experiment/unlearn/popqa/r2d_lora.yaml b/configs/experiment/unlearn/popqa/r2d_lora.yaml
index 36e3ce7..610f6c0 100644
--- a/configs/experiment/unlearn/popqa/r2d_lora.yaml
+++ b/configs/experiment/unlearn/popqa/r2d_lora.yaml
@@ -48,8 +48,10 @@ trainer:
     gradient_accumulation_steps: 32
     learning_rate: 1e-5
     num_train_epochs: 5
+    optim: sgd
+    weight_decay: 0.0
     lr_scheduler_type: constant
-    warmup_ratio: 0.1
+    warmup_ratio: 0.0
     logging_steps: 10
     eval_strategy: "no"
     save_strategy: "no"
@@ -59,7 +61,7 @@ trainer:
     gradient_checkpointing: false
     ddp_find_unused_parameters: false
   method_args:
-    noise_std: 0.0
+    noise_std: null
     noise_trainable_only: true
 
 task_name: popqa_r2d_lora
diff --git a/configs/trainer/R2D.yaml b/configs/trainer/R2D.yaml
index 5be3b95..092314a 100644
--- a/configs/trainer/R2D.yaml
+++ b/configs/trainer/R2D.yaml
@@ -4,10 +4,21 @@ defaults:
 handler: R2D
 
 method_args:
-  noise_std: 0.0
-  noise_seed: null
+  # direct noise (if set, overrides DP params)
+  noise_std: null
+  noise_seed: 0
   noise_trainable_only: true
 
+  # DP calibration inputs (if noise_std is null)
   dp_epsilon: null
   dp_delta: null
   dp_sensitivity: null
+  dp_use_analytic_gaussian: true
+
+  # optional paper-based sensitivity when dp_sensitivity is null
+  r2d_L: null
+  r2d_G: null
+  r2d_n: null
+  r2d_m: null
+  r2d_rewind_step: null
+  r2d_eta: null
diff --git a/scripts/duet/r2d_duet.sh b/scripts/duet/r2d_duet.sh
index d3dcddb..2de4604 100755
--- a/scripts/duet/r2d_duet.sh
+++ b/scripts/duet/r2d_duet.sh
@@ -87,10 +87,36 @@ lora_rs=(${LORA_RS:-"32"})
 lora_alphas=(${LORA_ALPHAS:-"64"})
 lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
 
-r2d_noise_std="${R2D_NOISE_STD:-0.0}"
-r2d_noise_seed="${R2D_NOISE_SEED:-42}"
+r2d_noise_std="${R2D_NOISE_STD:-null}"
+r2d_noise_seed="${R2D_NOISE_SEED:-0}"
+r2d_eps="${R2D_EPS:-null}"
+r2d_delta="${R2D_DELTA:-null}"
+r2d_sens="${R2D_SENS:-null}"
+r2d_use_analytic_gaussian="${R2D_USE_ANALYTIC_GAUSSIAN:-true}"
+r2d_L="${R2D_L:-null}"
+r2d_G="${R2D_G:-null}"
+r2d_n="${R2D_N:-null}"
+r2d_m="${R2D_M:-null}"
+r2d_eta="${R2D_ETA:-null}"
+r2d_rewind_step_for_sigma="${R2D_REWIND_STEP_FOR_SIGMA:-}"
+if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
+    if [[ -n "${R2D_REWIND_STEP:-}" ]]; then
+        r2d_rewind_step_for_sigma="${R2D_REWIND_STEP}"
+    elif [[ "${rewind_subfolder}" =~ checkpoint-([0-9]+) ]]; then
+        r2d_rewind_step_for_sigma="${BASH_REMATCH[1]}"
+    elif [[ "${rewind_model_path}" =~ checkpoint-([0-9]+) ]]; then
+        r2d_rewind_step_for_sigma="${BASH_REMATCH[1]}"
+    fi
+fi
+if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
+    r2d_rewind_step_for_sigma="null"
+fi
 delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
 
+if [[ "${max_steps}" == "0" ]]; then
+    echo "[duet][R2D] WARNING: R2D_MAX_STEPS=0, using NUM_EPOCHS instead of explicit K steps."
+fi
+
 export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
 
 for split in "${forget_retain_splits[@]}"; do
@@ -105,7 +131,9 @@ for split in "${forget_retain_splits[@]}"; do
                 for lora_dropout in "${lora_dropouts[@]}"; do
                     dropout_tag=${lora_dropout//./p}
                     rewind_tag="${R2D_REWIND_TAG:-rewind}"
-                    task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${r2d_noise_std}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                    sigma_tag=${r2d_noise_std//./p}
+                    sigma_tag=${sigma_tag//null/dp}
+                    task_name=duet_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
                     run_dir=${output_root}/${task_name}
                     eval_dir=${run_dir}/evals
                     summary_path=${eval_dir}/DUET_SUMMARY.json
@@ -144,6 +172,8 @@ for split in "${forget_retain_splits[@]}"; do
                             trainer.args.num_train_epochs=${num_train_epochs} \
                             trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                             trainer.args.learning_rate=${lr} \
+                            trainer.args.optim=sgd \
+                            trainer.args.weight_decay=0.0 \
                             trainer.args.lr_scheduler_type=constant \
                             trainer.args.warmup_ratio=0.0 \
                             trainer.args.save_strategy=no \
@@ -154,6 +184,16 @@ for split in "${forget_retain_splits[@]}"; do
                             trainer.method_args.noise_std=${r2d_noise_std} \
                             trainer.method_args.noise_seed=${r2d_noise_seed} \
                             trainer.method_args.noise_trainable_only=true \
+                            trainer.method_args.dp_epsilon=${r2d_eps} \
+                            trainer.method_args.dp_delta=${r2d_delta} \
+                            trainer.method_args.dp_sensitivity=${r2d_sens} \
+                            trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
+                            trainer.method_args.r2d_L=${r2d_L} \
+                            trainer.method_args.r2d_G=${r2d_G} \
+                            trainer.method_args.r2d_n=${r2d_n} \
+                            trainer.method_args.r2d_m=${r2d_m} \
+                            trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
+                            trainer.method_args.r2d_eta=${r2d_eta} \
                             retain_logs_path=null \
                             "${extra_train_args[@]}" \
                             paths.output_dir=${run_dir} \
diff --git a/scripts/popqa/r2d_popqa.sh b/scripts/popqa/r2d_popqa.sh
index 0dd307c..9f5c054 100755
--- a/scripts/popqa/r2d_popqa.sh
+++ b/scripts/popqa/r2d_popqa.sh
@@ -98,10 +98,36 @@ lora_rs=(${LORA_RS:-"32"})
 lora_alphas=(${LORA_ALPHAS:-"64"})
 lora_dropouts=(${LORA_DROPOUTS:-"0.0"})
 
-r2d_noise_std="${R2D_NOISE_STD:-0.0}"
-r2d_noise_seed="${R2D_NOISE_SEED:-42}"
+r2d_noise_std="${R2D_NOISE_STD:-null}"
+r2d_noise_seed="${R2D_NOISE_SEED:-0}"
+r2d_eps="${R2D_EPS:-null}"
+r2d_delta="${R2D_DELTA:-null}"
+r2d_sens="${R2D_SENS:-null}"
+r2d_use_analytic_gaussian="${R2D_USE_ANALYTIC_GAUSSIAN:-true}"
+r2d_L="${R2D_L:-null}"
+r2d_G="${R2D_G:-null}"
+r2d_n="${R2D_N:-null}"
+r2d_m="${R2D_M:-null}"
+r2d_eta="${R2D_ETA:-null}"
+r2d_rewind_step_for_sigma="${R2D_REWIND_STEP_FOR_SIGMA:-}"
+if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
+    if [[ -n "${R2D_REWIND_STEP:-}" ]]; then
+        r2d_rewind_step_for_sigma="${R2D_REWIND_STEP}"
+    elif [[ "${rewind_subfolder}" =~ checkpoint-([0-9]+) ]]; then
+        r2d_rewind_step_for_sigma="${BASH_REMATCH[1]}"
+    elif [[ "${rewind_model_path}" =~ checkpoint-([0-9]+) ]]; then
+        r2d_rewind_step_for_sigma="${BASH_REMATCH[1]}"
+    fi
+fi
+if [[ -z "${r2d_rewind_step_for_sigma}" ]]; then
+    r2d_rewind_step_for_sigma="null"
+fi
 delete_model_safetensors_after_eval="${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-0}"
 
+if [[ "${max_steps}" == "0" ]]; then
+    echo "[popqa][R2D] WARNING: R2D_MAX_STEPS=0, using NUM_EPOCHS instead of explicit K steps."
+fi
+
 export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
 
 for split in "${forget_retain_splits[@]}"; do
@@ -116,7 +142,9 @@ for split in "${forget_retain_splits[@]}"; do
                 for lora_dropout in "${lora_dropouts[@]}"; do
                     dropout_tag=${lora_dropout//./p}
                     rewind_tag="${R2D_REWIND_TAG:-rewind}"
-                    task_name=popqa_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${r2d_noise_std}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
+                    sigma_tag=${r2d_noise_std//./p}
+                    sigma_tag=${sigma_tag//null/dp}
+                    task_name=popqa_${base_model}_${forget_label}_r2d_${rewind_tag}_lr${lr}_sigma${sigma_tag}_r${lora_r}_a${lora_alpha}_d${dropout_tag}
                     run_dir=${output_root}/${task_name}
                     eval_dir=${run_dir}/evals
                     summary_path=${eval_dir}/POPQA_SUMMARY.json
@@ -155,6 +183,8 @@ for split in "${forget_retain_splits[@]}"; do
                             trainer.args.num_train_epochs=${num_train_epochs} \
                             trainer.args.gradient_checkpointing=${gradient_checkpointing} \
                             trainer.args.learning_rate=${lr} \
+                            trainer.args.optim=sgd \
+                            trainer.args.weight_decay=0.0 \
                             trainer.args.lr_scheduler_type=constant \
                             trainer.args.warmup_ratio=0.0 \
                             trainer.args.save_strategy=no \
@@ -165,6 +195,16 @@ for split in "${forget_retain_splits[@]}"; do
                             trainer.method_args.noise_std=${r2d_noise_std} \
                             trainer.method_args.noise_seed=${r2d_noise_seed} \
                             trainer.method_args.noise_trainable_only=true \
+                            trainer.method_args.dp_epsilon=${r2d_eps} \
+                            trainer.method_args.dp_delta=${r2d_delta} \
+                            trainer.method_args.dp_sensitivity=${r2d_sens} \
+                            trainer.method_args.dp_use_analytic_gaussian=${r2d_use_analytic_gaussian} \
+                            trainer.method_args.r2d_L=${r2d_L} \
+                            trainer.method_args.r2d_G=${r2d_G} \
+                            trainer.method_args.r2d_n=${r2d_n} \
+                            trainer.method_args.r2d_m=${r2d_m} \
+                            trainer.method_args.r2d_rewind_step=${r2d_rewind_step_for_sigma} \
+                            trainer.method_args.r2d_eta=${r2d_eta} \
                             retain_logs_path=null \
                             "${extra_train_args[@]}" \
                             paths.output_dir=${run_dir} \
diff --git a/src/trainer/unlearn/r2d.py b/src/trainer/unlearn/r2d.py
index d42ad70..ed03f2f 100644
--- a/src/trainer/unlearn/r2d.py
+++ b/src/trainer/unlearn/r2d.py
@@ -10,9 +10,9 @@ from trainer.utils import _filter_model_inputs
 logger = logging.getLogger(__name__)
 
 
-def _default_sigma_from_dp(epsilon: float, delta: float, sensitivity: float) -> float:
+def _basic_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
     """
-    Dependency-free fallback for Gaussian mechanism calibration:
+    Classic (non-analytic) Gaussian mechanism calibration:
         sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
     """
     if epsilon <= 0:
@@ -26,8 +26,127 @@ def _default_sigma_from_dp(epsilon: float, delta: float, sensitivity: float) ->
     )
 
 
+def calibrate_analytic_gaussian_mechanism(
+    epsilon: float, delta: float, gs: float, tol: float = 1e-12
+) -> float:
+    """
+    Analytic Gaussian mechanism calibration (Balle & Wang, ICML 2018).
+    Returns sigma (stddev of Gaussian noise).
+    """
+    if epsilon <= 0:
+        raise ValueError(f"epsilon must be > 0, got {epsilon}")
+    if not (0.0 < delta < 1.0):
+        raise ValueError(f"delta must be in (0,1), got {delta}")
+    if gs < 0:
+        raise ValueError(f"gs (global sensitivity) must be >= 0, got {gs}")
+    if gs == 0:
+        return 0.0
+
+    def _phi(x: float) -> float:
+        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
+
+    def _log_phi(x: float) -> float:
+        t = torch.tensor(x, dtype=torch.float64)
+        return float(torch.special.log_ndtr(t))
+
+    def _case_a(eps: float, s: float) -> float:
+        a = math.sqrt(eps * s)
+        b = math.sqrt(eps * (s + 2.0))
+        log_term = eps + _log_phi(-b)
+        return _phi(a) - math.exp(log_term)
+
+    def _case_b(eps: float, s: float) -> float:
+        a = math.sqrt(eps * s)
+        b = math.sqrt(eps * (s + 2.0))
+        log_term = eps + _log_phi(-b)
+        return _phi(-a) - math.exp(log_term)
+
+    def _doubling_trick(predicate_stop, s_inf: float, s_sup: float):
+        while not predicate_stop(s_sup):
+            s_inf = s_sup
+            s_sup = 2.0 * s_inf
+        return s_inf, s_sup
+
+    def _binary_search(predicate_stop, predicate_left, s_inf: float, s_sup: float):
+        s_mid = s_inf + (s_sup - s_inf) / 2.0
+        while not predicate_stop(s_mid):
+            if predicate_left(s_mid):
+                s_sup = s_mid
+            else:
+                s_inf = s_mid
+            s_mid = s_inf + (s_sup - s_inf) / 2.0
+        return s_mid
+
+    delta_thr = _case_a(epsilon, 0.0)
+    if delta == delta_thr:
+        alpha = 1.0
+    else:
+        if delta > delta_thr:
+            predicate_stop_dt = lambda s: _case_a(epsilon, s) >= delta
+            f_s_to_delta = lambda s: _case_a(epsilon, s)
+            predicate_left_bs = lambda s: f_s_to_delta(s) > delta
+            f_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
+        else:
+            predicate_stop_dt = lambda s: _case_b(epsilon, s) <= delta
+            f_s_to_delta = lambda s: _case_b(epsilon, s)
+            predicate_left_bs = lambda s: f_s_to_delta(s) < delta
+            f_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)
+
+        predicate_stop_bs = lambda s: abs(f_s_to_delta(s) - delta) <= tol
+        s_inf, s_sup = _doubling_trick(predicate_stop_dt, 0.0, 1.0)
+        s_final = _binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
+        alpha = f_s_to_alpha(s_final)
+
+    return float(alpha * gs / math.sqrt(2.0 * epsilon))
+
+
+def _h_function_from_paper(
+    *,
+    K: int,
+    eta: float,
+    L: float,
+    m: int,
+    n: int,
+    rewind_step: int,
+) -> float:
+    """
+    Paper-defined h(K) helper with rewind_step = (T-K) checkpoint index.
+    """
+    if n <= m:
+        raise ValueError(f"Need n > m, got n={n}, m={m}")
+    if K < 0 or rewind_step < 0:
+        raise ValueError("K and rewind_step must be >= 0")
+    if L <= 0 or eta <= 0:
+        raise ValueError("L and eta must be > 0")
+
+    a = eta * L * n / (n - m)
+    b = eta * L
+
+    term1 = math.expm1(rewind_step * math.log1p(a))  # (1+a)^rewind_step - 1
+    term2 = math.exp(K * math.log1p(b))  # (1+b)^K
+    return float(term1 * term2)
+
+
+def _paper_sensitivity(
+    *,
+    K: int,
+    eta: float,
+    L: float,
+    G: float,
+    m: int,
+    n: int,
+    rewind_step: int,
+) -> float:
+    """
+    Paper-equivalent global sensitivity:
+      GS = 2 m G h(K) / (L n)
+    """
+    h = _h_function_from_paper(K=K, eta=eta, L=L, m=m, n=n, rewind_step=rewind_step)
+    return float((2.0 * m * G * h) / (L * n))
+
+
 @torch.no_grad()
-def _apply_noise(
+def _add_noise_to_params(
     model: torch.nn.Module,
     sigma: float,
     trainable_only: bool = True,
@@ -55,7 +174,7 @@ def add_gaussian_noise_to_weights(
         return
 
     if seed is None:
-        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)
+        _add_noise_to_params(model=model, sigma=sigma, trainable_only=trainable_only)
         return
 
     devices = []
@@ -66,7 +185,7 @@ def add_gaussian_noise_to_weights(
         torch.manual_seed(seed)
         if torch.cuda.is_available():
             torch.cuda.manual_seed_all(seed)
-        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)
+        _add_noise_to_params(model=model, sigma=sigma, trainable_only=trainable_only)
 
 
 class R2D(UnlearnTrainer):
@@ -79,58 +198,117 @@ class R2D(UnlearnTrainer):
     def __init__(
         self,
         *args,
-        noise_std: float = 0.0,
-        noise_seed: Optional[int] = None,
+        noise_std: Optional[float] = None,
+        noise_seed: int = 0,
         noise_trainable_only: bool = True,
         dp_epsilon: Optional[float] = None,
         dp_delta: Optional[float] = None,
         dp_sensitivity: Optional[float] = None,
+        dp_use_analytic_gaussian: bool = True,
+        r2d_L: Optional[float] = None,
+        r2d_G: Optional[float] = None,
+        r2d_n: Optional[int] = None,
+        r2d_m: Optional[int] = None,
+        r2d_rewind_step: Optional[int] = None,
+        r2d_eta: Optional[float] = None,
         **kwargs,
     ):
         super().__init__(*args, **kwargs)
-        self.noise_std = float(noise_std)
-        self.noise_seed = noise_seed
+        self.noise_std = None if noise_std is None else float(noise_std)
+        self.noise_seed = int(noise_seed)
         self.noise_trainable_only = bool(noise_trainable_only)
         self.dp_epsilon = dp_epsilon
         self.dp_delta = dp_delta
         self.dp_sensitivity = dp_sensitivity
+        self.dp_use_analytic_gaussian = bool(dp_use_analytic_gaussian)
+        self.r2d_L = r2d_L
+        self.r2d_G = r2d_G
+        self.r2d_n = r2d_n
+        self.r2d_m = r2d_m
+        self.r2d_rewind_step = r2d_rewind_step
+        self.r2d_eta = r2d_eta
         self._noise_applied = False
 
     def _resolve_sigma(self) -> float:
-        if self.noise_std < 0:
-            raise ValueError("noise_std must be >= 0")
-        if self.noise_std > 0:
+        if self.noise_std is not None:
+            if self.noise_std < 0:
+                raise ValueError("noise_std must be >= 0")
             return float(self.noise_std)
 
-        if (
-            self.dp_epsilon is not None
-            and self.dp_delta is not None
-            and self.dp_sensitivity is not None
+        if self.dp_epsilon is None or self.dp_delta is None:
+            return 0.0
+
+        eps = float(self.dp_epsilon)
+        delta = float(self.dp_delta)
+        gs = float(self.dp_sensitivity) if self.dp_sensitivity is not None else None
+
+        if gs is None and all(
+            v is not None
+            for v in (
+                self.r2d_L,
+                self.r2d_G,
+                self.r2d_n,
+                self.r2d_m,
+                self.r2d_rewind_step,
+            )
         ):
-            return _default_sigma_from_dp(
-                epsilon=float(self.dp_epsilon),
-                delta=float(self.dp_delta),
-                sensitivity=float(self.dp_sensitivity),
+            eta = (
+                float(self.r2d_eta)
+                if self.r2d_eta is not None
+                else float(getattr(self.args, "learning_rate", 0.0))
             )
+            K = int(getattr(self.state, "global_step", 0))
+            try:
+                gs = _paper_sensitivity(
+                    K=K,
+                    eta=eta,
+                    L=float(self.r2d_L),
+                    G=float(self.r2d_G),
+                    m=int(self.r2d_m),
+                    n=int(self.r2d_n),
+                    rewind_step=int(self.r2d_rewind_step),
+                )
+                logger.info("[R2D] Computed paper sensitivity GS=%s using K=%s.", gs, K)
+            except Exception as err:
+                logger.warning(
+                    "[R2D] Failed paper-based sensitivity computation (%s); falling back.",
+                    err,
+                )
 
-        return 0.0
+        if gs is None:
+            gs = 1.0
+            logger.warning(
+                "[R2D] No sensitivity provided; defaulting global sensitivity to 1.0."
+            )
+
+        if self.dp_use_analytic_gaussian:
+            return calibrate_analytic_gaussian_mechanism(epsilon=eps, delta=delta, gs=gs)
+        return _basic_gaussian_sigma(epsilon=eps, delta=delta, sensitivity=gs)
 
     def compute_loss(self, model, inputs, return_outputs=False):
-        retain_inputs = _filter_model_inputs(inputs["retain"])
-        outputs = model(**retain_inputs)
+        if isinstance(inputs, dict) and "retain" in inputs:
+            batch = inputs["retain"]
+        else:
+            batch = inputs
+
+        batch = _filter_model_inputs(batch)
+        outputs = model(**batch)
         loss = outputs.loss
         return (loss, outputs) if return_outputs else loss
 
     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
         out_dir = output_dir or self.args.output_dir
-        sigma = self._resolve_sigma()
+        if "checkpoint-" in str(out_dir):
+            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
 
-        if (not self._noise_applied) and sigma > 0 and "checkpoint-" not in str(out_dir):
-            seed = self.noise_seed
-            if seed is None:
-                seed = int(getattr(self.args, "seed", 42))
-            process_index = int(getattr(self.args, "process_index", 0))
-            effective_seed = int(seed) + process_index
+        if self._noise_applied:
+            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
+
+        if not self.is_world_process_zero():
+            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
+
+        sigma = self._resolve_sigma()
+        if sigma > 0:
             model_to_noise = self.model
             if getattr(self, "accelerator", None) is not None:
                 try:
@@ -142,14 +320,17 @@ class R2D(UnlearnTrainer):
                 "[R2D] Applying Gaussian output perturbation: sigma=%s trainable_only=%s seed=%s",
                 sigma,
                 self.noise_trainable_only,
-                effective_seed,
+                self.noise_seed,
             )
             add_gaussian_noise_to_weights(
                 model=model_to_noise,
                 sigma=sigma,
-                seed=effective_seed,
+                seed=self.noise_seed,
                 trainable_only=self.noise_trainable_only,
             )
-            self._noise_applied = True
+        else:
+            logger.info("[R2D] sigma=0; skipping output perturbation.")
+
+        self._noise_applied = True
 
         return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
```
