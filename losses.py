"""Original CE/Dice and intra-route prompt supervision, plus MT consistency."""

import math

import torch
from torch.nn import functional as F


def segmentation_loss(logits, target):
    target = target.long()
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(probabilities.dtype)
    axes = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(axes)
    denominator = probabilities.square().sum(axes) + one_hot.square().sum(axes)
    dice = 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    # Equivalent unweighted CE for valid class indices. PyTorch 2.0's fused
    # CUDA NLL reduction is nondeterministic; gather + mean supports strict mode.
    cross_entropy = -logits.log_softmax(1).gather(1, target.unsqueeze(1)).mean()
    return 0.4 * cross_entropy + 0.6 * dice


def rampup(epoch, length):
    if length <= 0:
        return 1.0
    fraction = min(max(epoch / length, 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - fraction) ** 2)


def feedback_loss(primary_logits, secondary_logits, cfg):
    """S2 -> S1 only. The mask and soft targets never receive gradients.

    Confidence is a heuristic, not ground truth. Report coverage even when zero.
    The mean is taken over selected pixels; a zero mask yields a finite zero.
    """
    with torch.no_grad():
        source = secondary_logits.softmax(1)
        source_confidence, source_class = source.max(1)
        primary_confidence, primary_class = primary_logits.detach().softmax(1).max(1)
        mask = (source_confidence >= cfg.feedback_confidence)
        mask &= source_confidence >= primary_confidence + cfg.feedback_margin
        if cfg.feedback_foreground_only:
            mask &= source_class != 0
        foreground = (source_class != 0) | (primary_class != 0)
        disagreement = source_class != primary_class
    per_pixel = F.kl_div(primary_logits.log_softmax(1), source, reduction="none").sum(1)
    selected = mask.sum()
    loss = (per_pixel * mask).sum() / selected.clamp_min(1)
    return loss, {
        "feedback_coverage": float(mask.float().mean()),
        "feedback_pixels": int(selected),
        "student_disagreement": float(disagreement.float().mean()),
        "foreground_disagreement": float((disagreement & foreground).sum() / foreground.sum().clamp_min(1)),
    }


def mean_teacher_loss(student, teacher, labels, labeled_mask, epoch, cfg,
                      feedback_ready=False, feedback_activated_epoch=None):
    """Teacher outputs correspond only to the unlabeled samples.

    Hard prompt targets are detached student outputs, as in the original FTP.
    Teacher-to-student soft consistency replaces bidirectional CPS.
    """
    labeled_mask = labeled_mask.bool()
    unlabeled_mask = ~labeled_mask
    if not labeled_mask.any() or not unlabeled_mask.any():
        raise ValueError("Every MT batch must contain labeled and unlabeled samples")
    supervised = student["lf_prompt"].new_zeros(())
    prompt_loss = supervised.clone()
    consistency = supervised.clone()
    branches = [branch for branch in ("lf", "hf") if f"{branch}_prompt" in student]
    branch_metrics = {}
    for branch in branches:
        predicted = student[f"{branch}_prompt"]
        branch_supervised = segmentation_loss(predicted[labeled_mask], labels[labeled_mask])
        supervised = supervised + branch_supervised
        branch_metrics[f"supervised_{branch}"] = float(branch_supervised.detach())
        # The independent two-student model has exactly one LF teacher target.
        teacher_key = f"{branch}_prompt" if cfg.architecture == "shared" else "lf_prompt"
        target = teacher[teacher_key].detach().softmax(dim=1)
        student_prob = predicted[unlabeled_mask].softmax(dim=1)
        consistency = consistency + (student_prob - target).square().sum(dim=1).mean()
        use_prompt = cfg.use_apgm if branch == "lf" else cfg.use_psgm
        if use_prompt:
            hard_target = predicted[unlabeled_mask].detach().argmax(dim=1)
            prompt_loss = prompt_loss + segmentation_loss(
                student[f"{branch}_plain"][unlabeled_mask], hard_target,
            )
    supervised_weight = max(1.0, 1.25 - 0.005 * epoch)
    prompt_weight = min(0.25, 0.005 * epoch) if cfg.use_apgm or cfg.use_psgm else 0.0
    consistency_weight = cfg.consistency_weight * rampup(epoch, cfg.consistency_rampup_epochs)
    feedback = supervised.new_zeros(())
    feedback_weight = 0.0
    feedback_metrics = {}
    if cfg.architecture == "dual":
        feedback, feedback_metrics = feedback_loss(student["lf_prompt"][unlabeled_mask],
                                                   student["hf_prompt"][unlabeled_mask], cfg)
        start = max(cfg.feedback_start_epoch, feedback_activated_epoch
                    if feedback_activated_epoch is not None else cfg.feedback_start_epoch)
        ready = not cfg.feedback_require_ready or (feedback_ready and feedback_activated_epoch is not None)
        if ready and epoch >= start:
            feedback_weight = cfg.feedback_weight * rampup(epoch - start, cfg.feedback_rampup_epochs)
    total = (supervised_weight * supervised + consistency_weight * consistency
             + prompt_weight * prompt_loss + feedback_weight * feedback)
    return total, {
        "loss": float(total.detach()), "supervised": float(supervised.detach()),
        "consistency": float(consistency.detach()), "prompt": float(prompt_loss.detach()),
        "consistency_weight": consistency_weight, "prompt_weight": prompt_weight,
        "feedback": float(feedback.detach()), "feedback_weight": feedback_weight,
        **branch_metrics, **feedback_metrics,
    }
