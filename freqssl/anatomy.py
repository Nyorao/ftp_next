"""Ordinal ambiguity routing for SPIDER's *relative* level labels.

These are research components, not a claim of anatomical L1/L5 identification.
The sequence model supplies evidence for supervision granularity, never a hard
renumbering rule. It allows arbitrary starting/ending levels and skipped levels.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


FAMILIES = ((0,), tuple(range(1, 10)), (10,), tuple(range(11, 20)))
FAMILY_IDS = (0,) + (1,) * 9 + (2,) + (3,) * 9


def family_probabilities(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 4 or probabilities.shape[1] != 20:
        raise ValueError("SPIDER probabilities must have shape (N,20,H,W)")
    return torch.stack([probabilities[:, ids].sum(1) for ids in FAMILIES], 1)


def family_log_probabilities(logits: Tensor) -> Tensor:
    logp = logits.float().log_softmax(1)
    return torch.stack([torch.logsumexp(logp[:, ids], 1) for ids in FAMILIES], 1)


def family_labels(labels: Tensor) -> Tensor:
    if labels.min() < 0 or labels.max() >= 20:
        raise ValueError("Only genuinely labeled SPIDER masks may be collapsed")
    return torch.tensor(FAMILY_IDS, device=labels.device)[labels.long()]


def monotone_marginals(emissions: Tensor, advance_penalty: float = 1.5,
                       skip_penalty: float = 2.0) -> Tensor:
    """Log-space forward/backward on a free-entry, free-exit monotone CRF.

    Input (N,T,K) contains log emission scores in caudal-to-cranial order.
    States can stay or increase; transitions that skip a state pay an additional
    cost. All starting and ending states are allowed, including a one-level FOV.
    Returns posterior state probabilities, not a best-path pseudo-label.
    """
    if emissions.ndim != 3 or min(emissions.shape) < 1:
        raise ValueError("Emissions must have nonempty shape (N,T,K)")
    if not torch.isfinite(emissions).all():
        raise ValueError("Nonfinite emissions")
    if any(not math.isfinite(v) or v < 0 for v in (advance_penalty, skip_penalty)):
        raise ValueError("Transition penalties must be finite and nonnegative")
    # Preserve double precision for the exhaustive numerical reference test.
    e = emissions if emissions.dtype == torch.float64 else emissions.float()
    states = torch.arange(e.shape[-1], device=e.device)
    jump = states[None, :] - states[:, None]
    transition = -advance_penalty * (jump > 0).to(e.dtype) - skip_penalty * (jump - 1).clamp_min(0)
    transition = transition.masked_fill(jump < 0, -torch.inf)
    forward = [e[:, 0]]
    for t in range(1, e.shape[1]):
        forward.append(e[:, t] + torch.logsumexp(forward[-1][:, :, None] + transition, 1))
    backward = [torch.zeros_like(e[:, -1])]
    for t in range(e.shape[1] - 2, -1, -1):
        backward.append(torch.logsumexp(transition + (e[:, t + 1] + backward[-1])[:, None, :], 2))
    logp = torch.stack(forward, 1) + torch.stack(backward[::-1], 1)
    return logp.softmax(-1)


@torch.no_grad()
def ordinal_evidence(probabilities: Tensor, bins: int = 56) -> Tensor:
    """Independent vertebra/disc chain marginals on the unchanged weak image.

    PNG rows are tensor axis -1 in the audited data. Row index increases towards
    the caudal end, so reverse it for the ascending relative-level chains.
    Empty/background rows carry weak emissions. Each tissue has its own chain:
    vertebral arches can overlap disc rows, so an alternating hard chain is unsafe.
    Background/canal get unit evidence; they have no relative-level ambiguity.
    """
    family_probabilities(probabilities)  # shape check
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >=2")
    p = probabilities.detach().float()
    profile = F.adaptive_avg_pool1d(p.mean(2), min(bins, p.shape[-1]))
    evidence = torch.ones_like(profile)
    for ids in (FAMILIES[1], FAMILIES[3]):
        scores = profile[:, ids]
        mass = scores.sum(1, keepdim=True)
        conditional = (scores + 1e-6) / (mass + len(ids) * 1e-6)
        # A thin or absent structure cannot impose a strong sequence decision.
        support = (mass / 0.05).clamp(0, 1)
        emissions = (conditional.log() * support).flip(-1).transpose(1, 2)
        posterior = monotone_marginals(emissions).transpose(1, 2).flip(-1)
        evidence[:, ids] = posterior
    return F.interpolate(evidence, size=p.shape[-1], mode="linear", align_corners=False)[:, :, None].expand_as(p)


@torch.no_grad()
def route_targets(peer: Tensor, own: Tensor, *, confidence: float = 0.8,
                  background_confidence: float = 0.95, background_weight: float = 0.25,
                  chain_threshold: float = 0.5, interval_mass: float = 0.8,
                  max_interval: int = 3, bins: int = 56, use_chain: bool = True) -> tuple[Tensor, dict]:
    """Choose singleton, short ordinal interval, tissue family, or abstention.

    Both teachers must agree on the tissue. Fine IDs additionally need two
    confident, agreeing teachers and independent chain support. Adjacent ID
    disagreement can supervise a bounded candidate interval if BOTH chains put
    enough mass there. Otherwise preserve only the agreed tissue membership.

    Output packs 20 binary allowed-class channels plus one pixel weight. Pack
    BEFORE CutMix and splice all 21 channels with the exact input rectangle.
    No sequence model is ever run on an anatomically spliced image.
    """
    if peer.shape != own.shape:
        raise ValueError("Teacher shapes must match")
    if any(not 0 <= x <= 1 for x in (confidence, background_confidence, background_weight,
                                     chain_threshold, interval_mass)):
        raise ValueError("Routing thresholds/weight must be in [0,1]")
    if type(max_interval) is not int or not 2 <= max_interval <= 9:
        raise ValueError("max_interval must be between 2 and 9")
    peer, own = peer.detach().float(), own.detach().float()
    pc, pf = family_probabilities(peer).max(1)
    oc, of = family_probabilities(own).max(1)
    threshold = torch.where(pf == 0, background_confidence, confidence)
    accepted = (pf == of) & (pc >= threshold) & (oc >= threshold)
    cls_family = torch.tensor(FAMILY_IDS, device=peer.device)[None, :, None, None]
    allowed = cls_family == pf[:, None]
    peer_conf, pi = peer.max(1)
    own_conf, oi = own.max(1)
    if use_chain:
        pe, oe = ordinal_evidence(peer, bins), ordinal_evidence(own, bins)
        supported = (pe.gather(1, pi[:, None])[:, 0] >= chain_threshold) & (oe.gather(1, oi[:, None])[:, 0] >= chain_threshold)
    else:
        pe = oe = None
        supported = torch.ones_like(accepted)
    numbered = (pf == 1) | (pf == 3)
    fine = accepted & (pi == oi) & (peer_conf >= threshold) & (own_conf >= threshold) & supported
    lower, upper = torch.minimum(pi, oi), torch.maximum(pi, oi)
    classes = torch.arange(20, device=peer.device)[None, :, None, None]
    interval = (classes >= lower[:, None]) & (classes <= upper[:, None]) & allowed
    within_family = (cls_family.expand_as(peer).gather(1, pi[:, None])[:, 0] == pf) & (cls_family.expand_as(own).gather(1, oi[:, None])[:, 0] == pf)
    narrow = accepted & numbered & within_family & (upper > lower) & ((upper - lower + 1) <= max_interval)
    narrow &= (peer.mul(interval).sum(1) >= interval_mass) & (own.mul(interval).sum(1) >= interval_mass)
    if use_chain:
        narrow &= (pe.mul(interval).sum(1) >= interval_mass) & (oe.mul(interval).sum(1) >= interval_mass)
    allowed = torch.where(narrow[:, None], interval, allowed)
    allowed = torch.where(fine[:, None], classes == pi[:, None], allowed)
    # Abstentions have a harmless all-class set (zero marginal loss), weight 0.
    allowed = torch.where(accepted[:, None], allowed, torch.ones_like(allowed))
    weight = accepted.float() * torch.where(pf == 0, background_weight, 1.0)
    stats = {"coverage": float(accepted.float().mean()),
             "foreground_coverage": float((accepted & (pf > 0)).float().mean()),
             "fine_coverage": float(fine.float().mean()),
             "interval_coverage": float(narrow.float().mean()),
             "family_only_coverage": float((accepted & ~fine & ~narrow).float().mean()),
             "abstention": float((~accepted).float().mean())}
    return torch.cat((allowed.float(), weight[:, None]), 1).detach(), stats


def marginal_set_loss(logits: Tensor, packed: Tensor) -> Tensor:
    """-log P(allowed class set), averaged over ALL pixels including rejected.

    Tissue-only supervision cannot invent a relative ID. A singleton reduces
    exactly to CE, and an all-class set yields zero. Teacher tensors stop grad.
    """
    if logits.ndim != 4 or logits.shape[1] != 20 or packed.shape != (logits.shape[0], 21, *logits.shape[2:]):
        raise ValueError("Expected 20 logits and 21 packed target channels")
    target = packed.detach()
    allowed, weight = target[:, :20] > 0.5, target[:, 20]
    if not allowed.any(1).all() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Every target needs a nonempty set and finite nonnegative weight")
    logp = logits.float().log_softmax(1)
    loss = -torch.logsumexp(logp.masked_fill(~allowed, -torch.inf), 1)
    return (loss * weight).mean()
