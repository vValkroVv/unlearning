import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return (
        nn.functional.kl_div(
            current_probs, ref_probs, reduction="batchmean", log_target=True
        ),
        outputs,
    )


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        if ref_model is None:
            disable_adapter = getattr(model, "disable_adapter", None)
            if disable_adapter is None:
                raise ValueError(
                    "Reference model is required for UNDIAL when adapters cannot be disabled."
                )
            with disable_adapter():
                teacher_logits = model(**inputs).logits
        else:
            teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Identify valid positions (ignore labels marked as -100)
    valid_mask = shift_labels != -100
    beta_tensor = torch.tensor(
        beta, dtype=shift_teacher_logits.dtype, device=shift_teacher_logits.device
    )

    if valid_mask.any():
        positions = valid_mask.nonzero(as_tuple=False)
        batch_indices = positions[:, 0]
        seq_indices = positions[:, 1]
        token_indices = shift_labels[valid_mask]
        shift_teacher_logits[batch_indices, seq_indices, token_indices] -= beta_tensor

    student_log_probs = F.log_softmax(shift_logits, dim=-1)
    teacher_log_probs = F.log_softmax(shift_teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()

    ce = -(teacher_probs * student_log_probs).sum(dim=-1)
    if valid_mask.any():
        ce = ce[valid_mask]
    return ce.float().mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def beta_from_pop_sum_tensor(pop_sum: torch.Tensor) -> torch.Tensor:
    """Compute dynamic beta from a tensor of pop_sum using the clipped power-law.

    beta(p) = clip(58.7 * p^(-0.796), 0.05, 2.0)

    Expects a float tensor; clamps p to avoid zero.
    Returns a tensor on the same device/dtype as input.
    """
    p = pop_sum.clamp(min=1e-8)
    beta_raw = 58.7 * torch.pow(p, -0.796)
    return beta_raw.clamp(min=0.05, max=2.0)


def _repetition_penalty_from_logits(shift_logits, shift_labels):
    """Compute an anti-repetition penalty based on probability of repeating the previous token.

    Uses teacher-forced labels as a proxy for previously generated token. For each position t>0,
    penalize the model's probability assigned at t to the previous ground-truth token y_{t-1}.
    Returns the mean penalty over valid positions.
    """
    # shift_logits: [B, T, V], shift_labels: [B, T]
    import torch.nn.functional as F

    B, T, V = shift_logits.shape
    if T <= 1:
        return torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    # current positions t=1..T-1
    cur_logits = shift_logits[:, 1:, :]  # [B, T-1, V]
    prev_labels = shift_labels[:, :-1]  # [B, T-1]
    valid = (prev_labels != -100) & (shift_labels[:, 1:] != -100)
    cur_log_probs = F.log_softmax(cur_logits, dim=-1)
    # gather log prob of previous label
    rep_logp = cur_log_probs.gather(-1, prev_labels.unsqueeze(-1)).squeeze(
        -1
    )  # [B, T-1]
    rep_prob = rep_logp.exp()
    if valid.any():
        rep_prob = rep_prob[valid]
        return rep_prob.mean()
    return torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs


def compute_wga_loss_dynamic_beta(
    model,
    inputs,
    beta_from_pop_sum: bool = True,
    rep_coeff: float = 0.0,
):
    """Compute WGA loss with per-sample dynamic beta and optional anti-repetition penalty.

    If `beta_from_pop_sum` and `inputs` contains a vector `pop_sum` of length B,
    per-sample beta is computed as:

        beta_i = clip(58.7 * (pop_sum[i]) ** (-0.796), beta_min, beta_max)

    where beta_min = 0.05 and beta_max = 2.0.
    The repetition penalty discourages repeating the previous token by penalizing
    p_t(y_{t-1}). Set `rep_coeff` > 0 to enable.
    """
    outputs = model(**{k: v for k, v in inputs.items() if k != "pop_sum"})
    labels = inputs["labels"].to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    B = shift_logits.size(0)
    T = shift_logits.size(1)

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    lm_loss = ce_loss(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    lm_loss = lm_loss.view(B, T)

    # Build beta per sample
    if beta_from_pop_sum and ("pop_sum" in inputs):
        pop_sum = inputs["pop_sum"].to(shift_logits.device).float().view(B)
        beta_vec = beta_from_pop_sum_tensor(pop_sum)
    else:
        beta_vec = torch.ones(B, device=shift_logits.device, dtype=shift_logits.dtype)

    # Broadcast beta to token dimension
    beta_bt = beta_vec.view(B, 1).expand(B, T)

    weight_ce = ((-lm_loss).exp().detach()) ** beta_bt
    valid = shift_labels != -100
    forget_loss = -(weight_ce[valid] * lm_loss[valid]).mean()

    if rep_coeff and rep_coeff > 0:
        rep_pen = _repetition_penalty_from_logits(shift_logits, shift_labels)
        forget_loss = forget_loss + rep_coeff * rep_pen
    return forget_loss, outputs


# end of file
