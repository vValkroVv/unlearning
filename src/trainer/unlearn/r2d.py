import logging
import math
from typing import Optional

import torch

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import _filter_model_inputs

logger = logging.getLogger(__name__)


def _default_sigma_from_dp(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Dependency-free fallback for Gaussian mechanism calibration:
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


@torch.no_grad()
def _apply_noise(
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
        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)
        return

    devices = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _apply_noise(model=model, sigma=sigma, trainable_only=trainable_only)


class R2D(UnlearnTrainer):
    """
    Rewind-to-Delete (R2D) unlearning:
    - train on retain-only NLL
    - apply optional Gaussian output perturbation on final save
    """

    def __init__(
        self,
        *args,
        noise_std: float = 0.0,
        noise_seed: Optional[int] = None,
        noise_trainable_only: bool = True,
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_sensitivity: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_std = float(noise_std)
        self.noise_seed = noise_seed
        self.noise_trainable_only = bool(noise_trainable_only)
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_sensitivity = dp_sensitivity
        self._noise_applied = False

    def _resolve_sigma(self) -> float:
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if self.noise_std > 0:
            return float(self.noise_std)

        if (
            self.dp_epsilon is not None
            and self.dp_delta is not None
            and self.dp_sensitivity is not None
        ):
            return _default_sigma_from_dp(
                epsilon=float(self.dp_epsilon),
                delta=float(self.dp_delta),
                sensitivity=float(self.dp_sensitivity),
            )

        return 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_inputs = _filter_model_inputs(inputs["retain"])
        outputs = model(**retain_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        out_dir = output_dir or self.args.output_dir
        sigma = self._resolve_sigma()

        if (not self._noise_applied) and sigma > 0 and "checkpoint-" not in str(out_dir):
            seed = self.noise_seed
            if seed is None:
                seed = int(getattr(self.args, "seed", 42))
            process_index = int(getattr(self.args, "process_index", 0))
            effective_seed = int(seed) + process_index
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
                effective_seed,
            )
            add_gaussian_noise_to_weights(
                model=model_to_noise,
                sigma=sigma,
                seed=effective_seed,
                trainable_only=self.noise_trainable_only,
            )
            self._noise_applied = True

        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
