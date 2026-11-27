"""Experimental tissue/relative-level advantage fusion for two frozen teachers.

The neutral point is the arithmetic mean of the original 20-class distributions.
Label-dependent advantage targets are a training/diagnostic operation only;
deployment and unlabeled target generation accept probabilities, not labels.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .anatomy import FAMILIES, family_labels, family_probabilities


def _pair(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    if (first.ndim != 4 or first.shape[1] != 20 or second.shape != first.shape
            or first.device != second.device or min(first.shape) < 1):
        raise ValueError("Expected two matching nonempty (N,20,H,W) probability tensors")
    dtype = torch.float64 if first.dtype == second.dtype == torch.float64 else torch.float32
    first, second = first.to(dtype), second.to(dtype)
    for p in (first, second):
        if (not torch.isfinite(p).all() or (p < 0).any()
                or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=2e-6, rtol=2e-6)):
            raise ValueError("Teacher probabilities must be finite, nonnegative and normalized")
    return first, second


def _divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    # An unsupported family stays zero. Adding epsilon would invent class mass
    # and break the neutral-mixture and unsupported-family invariants.
    return numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))


def _gates(gates: Tensor, reference: Tensor, count: int) -> Tensor:
    if gates.shape != (reference.shape[0], count, *reference.shape[2:]):
        raise ValueError(f"Expected {count} spatial gate channels")
    if not torch.isfinite(gates).all() or (gates.abs() > 20).any():
        raise ValueError("Gate log-odds must be finite and bounded by 20")
    return gates.to(reference)


def factor_fusion(first: Tensor, second: Tensor, gates: Tensor) -> Tensor:
    """Fuse tissue and within-tissue identity separately; positive favors first.

    gates = (tissue, vertebra, disc). The identity baseline uses ORIGINAL
    teacher tissue masses, independently of the tissue gate. Thus identity
    changes preserve tissue mass; tissue changes preserve conditional identity.
    All-zero gates reproduce the original arithmetic mean to numerical precision.
    """
    first, second = _pair(first, second)
    gates = _gates(gates, first, 3)
    a, b = family_probabilities(first), family_probabilities(second)
    tissue_weight = gates[:, :1].sigmoid()
    tissue = tissue_weight * a + (1 - tissue_weight) * b
    parts = []
    for family, ids in enumerate(FAMILIES):
        if len(ids) == 1:
            parts.append(tissue[:, family:family + 1])
            continue
        index = 1 if family == 1 else 2
        half = gates[:, index:index + 1] * .5
        numerator = half.exp() * first[:, ids] + (-half).exp() * second[:, ids]
        conditional = _divide(numerator, numerator.sum(1, keepdim=True))
        parts.append(tissue[:, family:family + 1] * conditional)
    return torch.cat(parts, 1)


def flat_fusion(first: Tensor, second: Tensor, gates: Tensor) -> Tensor:
    """Matched control: one spatial gate mixes the complete class distributions."""
    first, second = _pair(first, second)
    weight = _gates(gates, first, 1).sigmoid()
    return weight * first + (1 - weight) * second


@torch.no_grad()
def advantage_targets(first: Tensor, second: Tensor, labels: Tensor, *, mode="factor"):
    """Negative NLL gate gradients at the neutral mixture, using LABELED pixels.

    Returns bounded signed targets and masks. Positive means a small move toward
    teacher one decreases NLL at that pixel. These are local directions, not a
    guarantee that finite gate changes improve Dice or generalize to new images.
    Ignore targets (-100) are deliberately rejected: no unlabeled GT interface.
    """
    first, second = _pair(first, second)
    if labels.shape != first[:, 0].shape or labels.dtype != torch.long:
        raise ValueError("Expected integer labeled masks matching the probabilities")
    families = family_labels(labels)
    pa = first.gather(1, labels[:, None])
    pb = second.gather(1, labels[:, None])
    if mode == "flat":
        return .5 * _divide(pa - pb, pa + pb), torch.ones_like(pa, dtype=torch.bool)
    if mode != "factor":
        raise ValueError("mode must be factor or flat")
    a, b = family_probabilities(first), family_probabilities(second)
    ay, by = a.gather(1, families[:, None]), b.gather(1, families[:, None])
    targets = [.5 * _divide(ay - by, ay + by)]
    masks = [torch.ones_like(pa, dtype=torch.bool)]
    fine_first, fine_second = _divide(pa, pa + pb), _divide(pb, pa + pb)
    for group in (1, 3):
        total = a[:, group:group + 1] + b[:, group:group + 1]
        mass_first = _divide(a[:, group:group + 1], total)
        mass_second = _divide(b[:, group:group + 1], total)
        # Algebraically r(1-r)(b_A(y)-b_B(y))/b_0(y), avoiding products
        # of tiny probabilities in the denominator.
        direction = mass_second * fine_first - mass_first * fine_second
        mask = families[:, None] == group
        targets.append(torch.where(mask, direction, torch.zeros_like(direction)))
        masks.append(mask)
    return torch.cat(targets, 1), torch.cat(masks, 1)


def advantage_loss(prediction: Tensor, targets: Tensor, masks: Tensor,
                   labels: Tensor, background_weight=.25) -> Tensor:
    """Regress each available factor's local gain; only the router gets gradients."""
    if prediction.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError("Advantage predictions, targets and masks must match")
    if not 0 <= background_weight <= 1:
        raise ValueError("background_weight must be in [0,1]")
    weights = masks.detach().to(prediction.dtype)
    background = torch.where(labels == 0, background_weight, 1.).to(prediction)
    weights = weights * background[:, None]
    totals = weights.sum((0, 2, 3))
    errors = ((prediction - targets.detach()).square() * weights).sum((0, 2, 3))
    active = totals > 0
    per_factor = errors / totals.clamp_min(1)
    return (per_factor * active).sum() / active.sum().clamp_min(1)


@torch.no_grad()
def _descriptors(probabilities: Tensor) -> Tensor:
    tissue = family_probabilities(probabilities)
    entropies = []
    for group in (1, 3):
        conditional = _divide(probabilities[:, FAMILIES[group]], tissue[:, group:group + 1])
        entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(1, keepdim=True)
        entropies.append(entropy / math.log(9))
    return torch.cat((probabilities, tissue, *entropies), 1)


class AdvantageRouter(nn.Module):
    """Small antisymmetric predictor from detached local/axial teacher outputs.

    Uses no image, patient identifier, coordinate, mask or teacher identity bit.
    Swapping teachers negates every score by construction. The last layer is
    zero initialized, so a new router starts at the neutral mixture.
    """

    def __init__(self, mode="factor", width=16):
        super().__init__()
        if mode not in ("factor", "flat") or width < 4 or width % 4:
            raise ValueError("Expected mode factor/flat and width divisible by four")
        self.mode = mode
        self.local = nn.Sequential(nn.Conv2d(52, width, 1), nn.GroupNorm(4, width), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1, groups=width),
                                   nn.GroupNorm(4, width), nn.GELU())
        self.axial = nn.Conv1d(width, width, 7, dilation=4, padding=12)
        self.head = nn.Conv2d(width, 3 if mode == "factor" else 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        first, second = _pair(first.detach(), second.detach())
        a, b = _descriptors(first), _descriptors(second)
        ordered = torch.cat((torch.cat((a, b), 1), torch.cat((b, a), 1)), 0)
        features = self.local(ordered)
        context = self.axial(features.mean(2))[:, :, None]
        out = self.head(F.gelu(features + context))
        forward, reverse = out.chunk(2, 0)
        return ((forward - reverse) * .5).tanh()


@torch.no_grad()
def pack_soft_targets(first: Tensor, second: Tensor, fused: Tensor, *, confidence=.8,
                      background_confidence=.95, background_weight=.25) -> Tensor:
    """Use neutral-ensemble TISSUE confidence for the common selection control.

    Do not threshold fine-class confidence: that would discard the very level
    disagreements the soft target retains. Pack the weight WITH the complete
    probability distribution before the student's aligned CutMix.
    """
    first, second = _pair(first, second)
    _pair(fused, first)
    if any(not 0 <= v <= 1 for v in (confidence, background_confidence, background_weight)):
        raise ValueError("Confidence thresholds and background weight must be in [0,1]")
    certainty, family = family_probabilities((first + second) * .5).max(1)
    threshold = torch.where(family == 0, background_confidence, confidence)
    weight = (certainty >= threshold) * torch.where(family == 0, background_weight, 1.)
    return torch.cat((fused.detach(), weight[:, None]), 1)


def soft_consistency_loss(logits: Tensor, packed: Tensor) -> Tensor:
    """Full 20-class soft CE, weighted and averaged over all spatial positions."""
    if logits.ndim != 4 or logits.shape[1] != 20 or packed.shape != (logits.shape[0], 21, *logits.shape[2:]):
        raise ValueError("Expected 20 logits and 21 packed target channels")
    target = packed.detach()
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Soft probabilities and weights must be finite and nonnegative")
    if not torch.allclose(target[:, :20].sum(1), torch.ones_like(target[:, 20]), atol=2e-6, rtol=2e-6):
        raise ValueError("Soft targets must sum to one")
    return (-(target[:, :20] * logits.float().log_softmax(1)).sum(1) * target[:, 20]).mean()
