import logging
import math
from typing import Optional

import torch

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import _filter_model_inputs

logger = logging.getLogger(__name__)


def _basic_gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Classic (non-analytic) Gaussian mechanism calibration:
        sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if sensitivity < 0:
        raise ValueError("sensitivity must be >= 0")
    return float(sensitivity) * math.sqrt(2.0 * math.log(1.25 / float(delta))) / float(
        epsilon
    )


def calibrate_analytic_gaussian_mechanism(
    epsilon: float, delta: float, gs: float, tol: float = 1e-12
) -> float:
    """
    Analytic Gaussian mechanism calibration (Balle & Wang, ICML 2018).
    Returns sigma (stddev of Gaussian noise).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    if gs < 0:
        raise ValueError(f"gs (global sensitivity) must be >= 0, got {gs}")
    if gs == 0:
        return 0.0

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _log_phi(x: float) -> float:
        t = torch.tensor(x, dtype=torch.float64)
        return float(torch.special.log_ndtr(t))

    def _case_a(eps: float, s: float) -> float:
        a = math.sqrt(eps * s)
        b = math.sqrt(eps * (s + 2.0))
        log_term = eps + _log_phi(-b)
        return _phi(a) - math.exp(log_term)

    def _case_b(eps: float, s: float) -> float:
        a = math.sqrt(eps * s)
        b = math.sqrt(eps * (s + 2.0))
        log_term = eps + _log_phi(-b)
        return _phi(-a) - math.exp(log_term)

    def _doubling_trick(predicate_stop, s_inf: float, s_sup: float):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def _binary_search(predicate_stop, predicate_left, s_inf: float, s_sup: float):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = _case_a(epsilon, 0.0)
    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_dt = lambda s: _case_a(epsilon, s) >= delta
            f_s_to_delta = lambda s: _case_a(epsilon, s)
            predicate_left_bs = lambda s: f_s_to_delta(s) > delta
            f_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
        else:
            predicate_stop_dt = lambda s: _case_b(epsilon, s) <= delta
            f_s_to_delta = lambda s: _case_b(epsilon, s)
            predicate_left_bs = lambda s: f_s_to_delta(s) < delta
            f_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)

        predicate_stop_bs = lambda s: abs(f_s_to_delta(s) - delta) <= tol
        s_inf, s_sup = _doubling_trick(predicate_stop_dt, 0.0, 1.0)
        s_final = _binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
        alpha = f_s_to_alpha(s_final)

    return float(alpha * gs / math.sqrt(2.0 * epsilon))


def _h_function_from_paper(
    *,
    K: int,
    eta: float,
    L: float,
    m: int,
    n: int,
    rewind_step: int,
) -> float:
    """
    Paper-defined h(K) helper with rewind_step = (T-K) checkpoint index.
    """
    if n <= m:
        raise ValueError(f"Need n > m, got n={n}, m={m}")
    if K < 0 or rewind_step < 0:
        raise ValueError("K and rewind_step must be >= 0")
    if L <= 0 or eta <= 0:
        raise ValueError("L and eta must be > 0")

    a = eta * L * n / (n - m)
    b = eta * L

    term1 = math.expm1(rewind_step * math.log1p(a))  # (1+a)^rewind_step - 1
    term2 = math.exp(K * math.log1p(b))  # (1+b)^K
    return float(term1 * term2)


def _paper_sensitivity(
    *,
    K: int,
    eta: float,
    L: float,
    G: float,
    m: int,
    n: int,
    rewind_step: int,
) -> float:
    """
    Paper-equivalent global sensitivity:
      GS = 2 m G h(K) / (L n)
    """
    h = _h_function_from_paper(K=K, eta=eta, L=L, m=m, n=n, rewind_step=rewind_step)
    return float((2.0 * m * G * h) / (L * n))


@torch.no_grad()
def _add_noise_to_params(
    model: torch.nn.Module,
    sigma: float,
    trainable_only: bool = True,
) -> None:
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        if not torch.is_floating_point(param):
            continue
        noise = torch.randn_like(param, dtype=torch.float32) * sigma
        param.add_(noise.to(dtype=param.dtype))


@torch.no_grad()
def add_gaussian_noise_to_weights(
    model: torch.nn.Module,
    sigma: float,
    seed: Optional[int] = None,
    trainable_only: bool = True,
) -> None:
    """
    Output perturbation: add N(0, sigma^2) noise to model parameters.
    """
    if sigma <= 0:
        return

    if seed is None:
        _add_noise_to_params(model=model, sigma=sigma, trainable_only=trainable_only)
        return

    devices = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _add_noise_to_params(model=model, sigma=sigma, trainable_only=trainable_only)


class R2D(UnlearnTrainer):
    """
    Rewind-to-Delete (R2D) unlearning:
    - train on retain-only NLL
    - apply optional Gaussian output perturbation on final save
    """

    def __init__(
        self,
        *args,
        noise_std: Optional[float] = None,
        noise_seed: int = 0,
        noise_trainable_only: bool = True,
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity: Optional[float] = None,
        dp_use_analytic_gaussian: bool = True,
        r2d_L: Optional[float] = None,
        r2d_G: Optional[float] = None,
        r2d_n: Optional[int] = None,
        r2d_m: Optional[int] = None,
        r2d_rewind_step: Optional[int] = None,
        r2d_eta: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_std = None if noise_std is None else float(noise_std)
        self.noise_seed = int(noise_seed)
        self.noise_trainable_only = bool(noise_trainable_only)
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_sensitivity = dp_sensitivity
        self.dp_use_analytic_gaussian = bool(dp_use_analytic_gaussian)
        self.r2d_L = r2d_L
        self.r2d_G = r2d_G
        self.r2d_n = r2d_n
        self.r2d_m = r2d_m
        self.r2d_rewind_step = r2d_rewind_step
        self.r2d_eta = r2d_eta
        self._noise_applied = False

    def _resolve_sigma(self) -> float:
        if self.noise_std is not None:
            if self.noise_std < 0:
                raise ValueError("noise_std must be >= 0")
            return float(self.noise_std)

        if self.dp_epsilon is None and self.dp_delta is None:
            return 0.0
        if self.dp_epsilon is None or self.dp_delta is None:
            raise ValueError(
                "noise_std is null but only one of dp_epsilon/dp_delta is set; set both or neither."
            )

        eps = float(self.dp_epsilon)
        delta = float(self.dp_delta)
        gs = float(self.dp_sensitivity) if self.dp_sensitivity is not None else None

        if gs is None and all(
            v is not None
            for v in (
                self.r2d_L,
                self.r2d_G,
                self.r2d_n,
                self.r2d_m,
                self.r2d_rewind_step,
            )
        ):
            eta = (
                float(self.r2d_eta)
                if self.r2d_eta is not None
                else float(getattr(self.args, "learning_rate", 0.0))
            )
            K = int(getattr(self.state, "global_step", 0))
            try:
                gs = _paper_sensitivity(
                    K=K,
                    eta=eta,
                    L=float(self.r2d_L),
                    G=float(self.r2d_G),
                    m=int(self.r2d_m),
                    n=int(self.r2d_n),
                    rewind_step=int(self.r2d_rewind_step),
                )
                logger.info("[R2D] Computed paper sensitivity GS=%s using K=%s.", gs, K)
            except Exception as err:
                raise ValueError(
                    "R2D DP-mode requested (noise_std=null) and paper-based sensitivity "
                    f"computation failed: {err}. Set trainer.method_args.dp_sensitivity "
                    "or fix r2d_L/r2d_G/r2d_n/r2d_m/r2d_rewind_step/r2d_eta."
                ) from err

        if gs is None:
            raise ValueError(
                "R2D DP-mode requested (noise_std=null) but sensitivity is undefined. "
                "Set trainer.method_args.dp_sensitivity or provide paper inputs: "
                "r2d_L, r2d_G, r2d_n, r2d_m, r2d_rewind_step (and optionally r2d_eta)."
            )

        if self.dp_use_analytic_gaussian:
            return calibrate_analytic_gaussian_mechanism(epsilon=eps, delta=delta, gs=gs)
        return _basic_gaussian_sigma(epsilon=eps, delta=delta, sensitivity=gs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if isinstance(inputs, dict) and "retain" in inputs:
            batch = inputs["retain"]
        else:
            batch = inputs

        batch = _filter_model_inputs(batch)
        outputs = model(**batch)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        out_dir = output_dir or self.args.output_dir
        if "checkpoint-" in str(out_dir):
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if self._noise_applied:
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        if not self.is_world_process_zero():
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        sigma = self._resolve_sigma()
        if sigma > 0:
            model_to_noise = self.model
            if getattr(self, "accelerator", None) is not None:
                try:
                    model_to_noise = self.accelerator.unwrap_model(self.model)
                except Exception:
                    model_to_noise = self.model

            logger.info(
                "[R2D] Applying Gaussian output perturbation: sigma=%s trainable_only=%s seed=%s",
                sigma,
                self.noise_trainable_only,
                self.noise_seed,
            )
            add_gaussian_noise_to_weights(
                model=model_to_noise,
                sigma=sigma,
                seed=self.noise_seed,
                trainable_only=self.noise_trainable_only,
            )
        else:
            logger.info("[R2D] sigma=0; skipping output perturbation.")

        self._noise_applied = True

        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
