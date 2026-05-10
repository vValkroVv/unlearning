from __future__ import annotations

from typing import Any

import torch

from trainer.unlearn.grad_diff import GradDiff


class STAT(GradDiff):
    """
    Synthetic Token Alternative Training baseline.

    STAT replaces only supervised forget-answer token positions with uniformly
    sampled vocabulary IDs, then trains CE on those synthetic answer tokens plus
    the normal retain CE/KL branch inherited from GradDiff.
    """

    def __init__(
        self,
        stat_forget_weight: float = 1.0,
        stat_retain_weight: float = 1.0,
        synthetic_mode: str = "uniform",
        exclude_special_tokens: bool = True,
        preserve_eos: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        # Accept GradDiff-style aliases so campaign wrappers can reuse alpha /
        # gamma conventions without changing STAT's explicit config names.
        alpha = kwargs.pop("alpha", None)
        gamma = kwargs.pop("gamma", None)
        if alpha is not None and stat_retain_weight == 1.0:
            stat_retain_weight = alpha
        if gamma is not None and stat_forget_weight == 1.0:
            stat_forget_weight = gamma

        super().__init__(
            alpha=float(stat_retain_weight),
            gamma=float(stat_forget_weight),
            *args,
            **kwargs,
        )
        self.stat_forget_weight = float(stat_forget_weight)
        self.stat_retain_weight = float(stat_retain_weight)
        self.synthetic_mode = str(synthetic_mode)
        self.exclude_special_tokens = bool(exclude_special_tokens)
        self.preserve_eos = bool(preserve_eos)
        self._valid_token_cache: dict[tuple[str, int, tuple[int, ...]], torch.Tensor] = {}

        if self.synthetic_mode != "uniform":
            raise ValueError(
                "STAT currently supports synthetic_mode='uniform' only; "
                f"got {self.synthetic_mode!r}."
            )

    @staticmethod
    def _as_model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        # Some counterfactual batches are nested as {"original": ...}; normal
        # artifact-free QA batches already expose tensors at the top level.
        if isinstance(batch, dict) and "original" in batch:
            batch = batch["original"]
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _tokenizer_special_ids(self) -> set[int]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            return set()

        special_ids: set[int] = set()
        for value in getattr(tokenizer, "all_special_ids", []) or []:
            if isinstance(value, int):
                special_ids.add(value)

        for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, int):
                special_ids.add(value)
        return special_ids

    def _eos_token_id(self) -> int | None:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self, "processing_class", None)
        value = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        return value if isinstance(value, int) else None

    @staticmethod
    def _vocab_size(model: torch.nn.Module) -> int:
        embedding = None
        try:
            embedding = model.get_input_embeddings()
        except Exception:
            embedding = None
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            return int(embedding.num_embeddings)

        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Could not infer vocabulary size for STAT synthetic sampling.")
        return int(vocab_size)

    def _valid_token_ids(self, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
        vocab_size = self._vocab_size(model)
        special_ids = self._tokenizer_special_ids() if self.exclude_special_tokens else set()
        bounded_special_ids = tuple(sorted(idx for idx in special_ids if 0 <= idx < vocab_size))
        key = (str(device), vocab_size, bounded_special_ids)

        cached = self._valid_token_cache.get(key)
        if cached is not None:
            return cached

        valid = torch.ones(vocab_size, dtype=torch.bool)
        if bounded_special_ids:
            valid[list(bounded_special_ids)] = False
        valid_ids = valid.nonzero(as_tuple=False).flatten().to(device=device, dtype=torch.long)
        if valid_ids.numel() == 0:
            raise ValueError("No valid token ids remain after excluding special tokens.")

        self._valid_token_cache[key] = valid_ids
        return valid_ids

    def _sample_uniform_token_ids(
        self,
        model: torch.nn.Module,
        shape: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        valid_ids = self._valid_token_ids(model, device)
        offsets = torch.randint(
            low=0,
            high=valid_ids.numel(),
            size=shape,
            device=device,
        )
        return valid_ids[offsets]

    def _make_synthetic_forget_inputs(
        self,
        model: torch.nn.Module,
        forget_inputs: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        input_ids = forget_inputs["input_ids"].clone()
        labels = forget_inputs["labels"].clone()
        attention_mask = forget_inputs["attention_mask"]

        target_mask = labels.ne(-100)
        if self.preserve_eos:
            eos_id = self._eos_token_id()
            if eos_id is not None:
                target_mask = target_mask & labels.ne(eos_id)

        if target_mask.any():
            synthetic_ids = self._sample_uniform_token_ids(
                model=model,
                shape=labels.shape,
                device=labels.device,
            )
            input_ids[target_mask] = synthetic_ids[target_mask]
            labels[target_mask] = synthetic_ids[target_mask]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }, target_mask

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = self._as_model_inputs(inputs["forget"])
        stat_forget_inputs, target_mask = self._make_synthetic_forget_inputs(
            model=model,
            forget_inputs=forget_inputs,
        )
        forget_outputs = model(**stat_forget_inputs)
        stat_forget_loss = forget_outputs.loss

        retain_inputs = self._as_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.stat_forget_weight * stat_forget_loss
            + self.stat_retain_weight * retain_loss
        )

        try:
            target_tokens_per_sample = target_mask.sum(dim=-1).float().mean()
            self.log(
                {
                    "stat_forget_loss": float(stat_forget_loss.detach().item()),
                    "stat_retain_loss": float(retain_loss.detach().item()),
                    "stat_total_loss": float(loss.detach().item()),
                    "stat_forget_weight": self.stat_forget_weight,
                    "stat_retain_weight": self.stat_retain_weight,
                    "stat_target_tokens_mean": float(target_tokens_per_sample.detach().item()),
                    "stat_exclude_special_tokens": 1.0 if self.exclude_special_tokens else 0.0,
                    "stat_preserve_eos": 1.0 if self.preserve_eos else 0.0,
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss
