import math

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


def _as_model_inputs(batch):
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }


def compute_unilogit_loss(
    model,
    inputs,
    forget_coef: float = 1.0,
    kl_direction: str = "model_to_target",
):
    """Compute the Unilogit uniform-target self-distillation loss.

    The implementation follows the released Unilogit code path but fixes two
    repo-fit issues:

    1. `attention_mask` is passed into the model, which is required for padded
       QA batches.
    2. ignored labels (`-100`) are filtered before target-logit assignment, so
       they are never used as negative tensor indices.

    `kl_direction=model_to_target` matches the uploaded Unilogit trainer call:
    `F.kl_div(soft_targets, soft_outputs, log_target=True)`, i.e. KL(current
    model distribution || modified uniform-target distribution).
    """

    model_inputs = _as_model_inputs(inputs)
    outputs = model(**model_inputs)

    logits = outputs.logits
    labels = model_inputs["labels"].to(device=logits.device)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels != -100

    if not valid_mask.any():
        return logits.new_zeros((), dtype=torch.float32), outputs

    vocab_size = shift_logits.size(-1)
    if vocab_size <= 1:
        raise ValueError("Unilogit requires vocab_size > 1.")

    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    flat_valid = valid_mask.view(-1)

    valid_logits = flat_logits[flat_valid]
    valid_labels = flat_labels[flat_valid].long()

    # The target distribution is detached from the graph: self-distillation
    # target comes from the current logits after replacing the gold-token logit.
    target_logits = valid_logits.detach().clone()
    row_indices = torch.arange(
        target_logits.size(0),
        device=target_logits.device,
    )

    target_logits[row_indices, valid_labels] = float("-inf")
    log_v_minus_one = math.log(vocab_size - 1)
    uniform_label_logits = torch.logsumexp(target_logits, dim=-1) - log_v_minus_one
    target_logits[row_indices, valid_labels] = uniform_label_logits.to(target_logits.dtype)

    model_log_probs = F.log_softmax(valid_logits, dim=-1)
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    if kl_direction == "model_to_target":
        token_kl = F.kl_div(
            target_log_probs,
            model_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    elif kl_direction == "target_to_model":
        token_kl = F.kl_div(
            model_log_probs,
            target_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    else:
        raise ValueError(
            "kl_direction must be one of {'model_to_target', 'target_to_model'}, "
            f"got {kl_direction!r}."
        )

    return token_kl.float().mean() * float(forget_coef), outputs


class Unilogit(GradDiff):
    def __init__(
        self,
        forget_coef: float = 1.0,
        kl_direction: str = "model_to_target",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.forget_coef = float(forget_coef)
        self.kl_direction = str(kl_direction)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = _as_model_inputs(inputs["forget"])
        forget_loss, forget_outputs = compute_unilogit_loss(
            model=model,
            inputs=forget_inputs,
            forget_coef=self.forget_coef,
            kl_direction=self.kl_direction,
        )

        retain_inputs = _as_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        try:
            self.log(
                {
                    "unilogit_forget_loss": float(forget_loss.detach().item()),
                    "unilogit_retain_loss": float(retain_loss.detach().item()),
                    "unilogit_total_loss": float(loss.detach().item()),
                    "unilogit_forget_coef": self.forget_coef,
                    "unilogit_alpha": float(self.alpha),
                    "unilogit_gamma": float(self.gamma),
                }
            )
        except Exception:
            pass

        return (loss, forget_outputs) if return_outputs else loss


# Backwards-compatible spelling for older configs or local patches.
UniLogit = Unilogit
