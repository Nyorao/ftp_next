"""Supervised and confidence-masked cross-teacher objectives.

These small objectives support an experimental baseline inspired by mean
teacher and cross-pseudo supervision; they are not an official paper replica.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _check_logits(logits: Tensor) -> None:
    if logits.ndim != 4 or min(logits.shape) == 0 or logits.shape[1] < 2:
        raise ValueError("Logits must have nonempty shape (N, C, H, W) with C >= 2")


def segmentation_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Equal-weight cross entropy and squared-denominator soft Dice.

    Dice pools the batch and spatial dimensions, then averages over all classes,
    including background. Call only on genuinely labeled images: ignored or
    unlabeled target values are intentionally unsupported.
    """
    _check_logits(logits)
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("Labels must have shape (N, H, W) matching logits")
    if labels.is_floating_point() or labels.is_complex():
        raise ValueError("Labels must contain integer class indices")
    labels = labels.long()
    values = logits.float()
    # PyTorch 2.0 CUDA fused NLL reduction rejects strict determinism. This
    # equivalent CE keeps all computation on-device and uses valid class IDs.
    cross_entropy = -values.log_softmax(1).gather(1, labels[:, None]).mean()
    probabilities = values.softmax(dim=1)
    targets = F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.square().sum(dim=dimensions) + targets.square().sum(dim=dimensions)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 0.5 * cross_entropy + 0.5 * (1.0 - dice.mean())


def masked_pseudo_loss(
    logits: Tensor,
    teacher_probabilities: Tensor,
    confidence: float = 0.8,
    background_confidence: float = 0.95,
    background_weight: float = 0.25,
) -> tuple[Tensor, dict[str, float | int | list[int]]]:
    """Use detached hard pseudo labels, with stricter/background-weighted gating.

    The reduction denominator is *all* image pixels, including rejected ones.
    Thus a tiny accepted set cannot receive the same total weight as a complete
    pseudo-labeled image. An empty mask returns a differentiable zero.

    Coverage values all use the total pixel count as denominator, so foreground
    and background coverage sum to overall coverage. Counts and fractions refer
    to the teacher's predictions, never to hidden ground truth.
    """
    _check_logits(logits)
    if teacher_probabilities.shape != logits.shape:
        raise ValueError("Teacher probabilities and student logits must have identical shapes")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")
    if not math.isfinite(background_confidence) or not 0.0 <= background_confidence <= 1.0:
        raise ValueError("background_confidence must be finite and between 0 and 1")
    if not math.isfinite(background_weight) or background_weight < 0.0:
        raise ValueError("background_weight must be finite and nonnegative")

    teacher = teacher_probabilities.detach().float()
    certainty, pseudo_labels = teacher.max(dim=1)
    foreground = pseudo_labels != 0
    selected_foreground = foreground & (certainty >= confidence)
    selected_background = ~foreground & (certainty >= background_confidence)
    weights = selected_foreground.float() + background_weight * selected_background.float()
    per_pixel_loss = -logits.float().log_softmax(1).gather(1, pseudo_labels[:, None]).squeeze(1)
    loss = (per_pixel_loss * weights).mean()

    total_pixels = pseudo_labels.numel()
    # A single small scalar transfer for reporting; image tensors stay on-device.
    foreground_pixels, accepted_foreground, accepted_background = torch.stack(
        (foreground.sum(), selected_foreground.sum(), selected_background.sum())
    ).tolist()
    accepted_pixels = accepted_foreground + accepted_background
    selected = selected_foreground | selected_background
    # Equality/reduction also supports strict CUDA determinism on torch 2.0;
    # CUDA bincount is rejected even for these integer diagnostic counts.
    class_membership = pseudo_labels[..., None] == torch.arange(logits.shape[1], device=logits.device)
    per_class_selected = (class_membership & selected[..., None]).sum((0, 1, 2)).tolist()
    per_class_predicted = class_membership.sum((0, 1, 2)).tolist()
    metrics: dict[str, float | int | list[int]] = {
        "coverage": accepted_pixels / total_pixels,
        "foreground_coverage": accepted_foreground / total_pixels,
        "background_coverage": accepted_background / total_pixels,
        "selected_pixels": accepted_pixels,
        "selected_foreground_pixels": accepted_foreground,
        "selected_background_pixels": accepted_background,
        "predicted_foreground_fraction": foreground_pixels / total_pixels,
        "selected_pixels_per_class": per_class_selected,
        "predicted_pixels_per_class": per_class_predicted,
    }
    return loss, metrics


def linear_ramp(step: int, warmup: int, ramp_steps: int) -> float:
    """Remain zero before warmup, then linearly reach one over ramp_steps."""
    if warmup < 0 or ramp_steps < 0:
        raise ValueError("warmup and ramp_steps must be nonnegative")
    if step < warmup:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - warmup) / ramp_steps))


def masked_pseudo_dice_loss(
    logits: Tensor,
    teacher_probabilities: Tensor,
    confidence: float = 0.8,
    background_confidence: float = 0.95,
    background_weight: float = 0.25,
) -> tuple[Tensor, dict[str, float | int]]:
    """Cross-Teaching-inspired macro foreground Dice on trusted pseudo labels.

    Apply the same pixel weights to intersection AND both denominator terms.
    Average only foreground classes with accepted targets. Each class's loss is
    attenuated by accepted / predicted target pixels, so a tiny confident subset
    of an otherwise uncertain class cannot receive full class weight. An empty
    foreground mask gives differentiable zero, never a background-only reward.
    This EMA/masked/foreground variant is not the published Cross Teaching loss.
    """
    _check_logits(logits)
    if teacher_probabilities.shape != logits.shape:
        raise ValueError("Teacher probabilities and student logits must have identical shapes")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (confidence, background_confidence)):
        raise ValueError("Confidence thresholds must be finite and between 0 and 1")
    if not math.isfinite(background_weight) or background_weight < 0:
        raise ValueError("background_weight must be finite and nonnegative")
    certainty, labels = teacher_probabilities.detach().float().max(1)
    foreground = labels != 0
    accepted_fg = foreground & (certainty >= confidence)
    accepted_bg = ~foreground & (certainty >= background_confidence)
    weights = (accepted_fg.float() + background_weight * accepted_bg.float())[:, None]
    targets = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()[:, 1:]
    probabilities = logits.float().softmax(1)[:, 1:]
    dims = (0, 2, 3)
    intersection = (weights * probabilities * targets).sum(dims)
    denominator = (weights * (probabilities.square() + targets)).sum(dims)
    per_class = 1.0 - (2 * intersection + 1e-5) / (denominator + 1e-5)
    selected = (accepted_fg[:, None] * targets).sum(dims)
    predicted = targets.sum(dims)
    reliability = selected / predicted.clamp_min(1)
    active = selected > 0
    count = active.sum().clamp_min(1)
    loss = (per_class * reliability * active).sum() / count
    return loss, {"pseudo_dice_classes": int(active.sum()),
                  "pseudo_dice_reliability": float(reliability.sum() / count)}


def multiscale_region_loss(
    student_features: Tensor,
    teacher_features: Tensor,
    grids: tuple[int, ...] = (4, 8, 16),
) -> tuple[Tensor, dict[str, float]]:
    """FRCNet-inspired spatial relations, adapted for independent students.

    Block-average the last decoder feature into g x g regions. L2-normalize
    channel descriptors before forming their spatial Gram matrix, then average
    MSE over pairs, images and scales. Unlike direct hidden-channel MSE, these
    relations are invariant to channel permutation / orthogonal basis changes.
    Cosine normalization, average pooling and smaller grids are adaptations, not
    an exact FRCNet reproduction. Targets always stop gradients.

    Inputs must describe the same image geometry. The caller uses UNMIXED strong
    student / weak teacher views; never splice already computed Gram matrices.
    """
    if student_features.ndim != 4 or teacher_features.ndim != 4:
        raise ValueError("Features must have shape (N, C, H, W)")
    if (student_features.shape[0] != teacher_features.shape[0]
            or student_features.shape[-2:] != teacher_features.shape[-2:]
            or min(student_features.shape) == 0 or min(teacher_features.shape) == 0):
        raise ValueError("Student and teacher must have aligned nonempty spatial features")
    height, width = student_features.shape[-2:]
    if (not grids or len(set(grids)) != len(grids)
            or any(type(g) is not int or not 2 <= g <= 32 or height % g or width % g for g in grids)):
        raise ValueError("Region grids must be unique spatial divisors between 2 and 32")
    target = teacher_features.detach().float()
    losses, metrics = [], {}
    for grid in grids:
        kernel = (height // grid, width // grid)
        # Fixed block pooling supports strict deterministic CUDA backward, unlike
        # adaptive_avg_pool2d on the repository's PyTorch 2.0 runtime.
        first = F.avg_pool2d(student_features.float(), kernel, stride=kernel).flatten(2)
        second = F.avg_pool2d(target, kernel, stride=kernel).flatten(2)
        first = F.normalize(first, dim=1, eps=1e-6)
        second = F.normalize(second, dim=1, eps=1e-6)
        relation = first.transpose(1, 2).bmm(first)
        reference = second.transpose(1, 2).bmm(second)
        loss = (relation - reference).square().mean()
        losses.append(loss)
        metrics[f"region_grid{grid}"] = float(loss.detach())
    return torch.stack(losses).mean(), metrics
