"""Experimental relative-ID transport for heterogeneous cross supervision.

The offset is inferred from predictions only. It aligns a peer to the receiving
teacher's numbering convention, which is NOT a ground-truth numbering oracle.
An out-of-range ID loses its mass; it never wraps or becomes an edge/background
label. Returned targets are deliberately sub-probabilities, without sharpening.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F

from .anatomy import FAMILY_IDS


def _pair(first: Tensor, second: Tensor):
    if (first.ndim != 4 or first.shape[1] != 20 or first.shape != second.shape
            or first.device != second.device or min(first.shape) < 1):
        raise ValueError('Expected matching nonempty N,20,H,W probabilities')
    first, second = first.detach().float(), second.detach().float()
    for p in (first, second):
        if (not torch.isfinite(p).all() or (p < 0).any()
                or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=2e-6, rtol=2e-6)):
            raise ValueError('Teachers must provide normalized finite probabilities')
    return first, second


@torch.no_grad()
def estimate_offset(first: Tensor, second: Tensor, *, confidence=.8,
                    minimum_pixels=16, minimum_support=.6, minimum_levels=2):
    """Return second-ID minus first-ID, with a symmetric abstention rule.

    Confident same-tissue correspondences form a 20x20 table per image. Each
    supported predicted class gets equal total vote, first from rows and then
    from columns. Averaging these directions makes swapping teachers reverse
    the offset. Vertebra and disc use one shared shift, but remain distinct
    classes. At least two different numerical levels must support a shift.
    Counts represent predicted classes, NOT connected anatomical instances.
    """
    a, b = _pair(first, second)
    if (not math.isfinite(confidence) or not 0 < confidence <= 1
            or not math.isfinite(minimum_support) or not .5 < minimum_support <= 1
            or type(minimum_pixels) is not int or minimum_pixels < 1
            or type(minimum_levels) is not int or not 2 <= minimum_levels <= 9):
        raise ValueError('Invalid confidence, vote support, pixel or level requirement')
    ac, ai = a.max(1); bc, bi = b.max(1)
    family = torch.tensor(FAMILY_IDS, device=a.device)
    numbered = (family[ai] == 1) | (family[ai] == 3)
    selected = numbered & (family[ai] == family[bi]) & (ac >= confidence) & (bc >= confidence)
    # Matmul of exact 0/1 values avoids nondeterministic CUDA bincount/scatter.
    ah = F.one_hot(ai.flatten(1), 20).float()
    bh = F.one_hot(bi.flatten(1), 20).float()
    counts = (ah * selected.flatten(1)[:, :, None]).transpose(1, 2) @ bh
    retained = ((counts.sum(2) >= minimum_pixels)[:, :, None]
                & (counts.sum(1) >= minimum_pixels)[:, None, :])
    counts = counts * retained
    rows, columns = counts.sum(2), counts.sum(1)
    row_vote = counts / rows.clamp_min(1)[:, :, None]
    row_vote = row_vote / (rows > 0).sum(1).clamp_min(1)[:, None, None]
    col_vote = counts / columns.clamp_min(1)[:, None, :]
    col_vote = col_vote / (columns > 0).sum(1).clamp_min(1)[:, None, None]
    vote = .5 * (row_vote + col_vote)
    ids = torch.arange(20, device=a.device)
    allowed = ((family[:, None] == family[None, :])
               & ((family[:, None] == 1) | (family[:, None] == 3)))
    delta = ids[None, :] - ids[:, None]
    candidates = torch.arange(-8, 9, device=a.device)
    masks = allowed[None] & (delta[None] == candidates[:, None, None])
    scores = (vote[:, None] * masks[None]).sum((2, 3))
    support, index = scores.max(1)
    proposed = candidates[index]
    matching = (counts >= minimum_pixels) & allowed & (delta[None] == proposed[:, None, None])
    supported_class = matching.any(2)
    # Vertebra 1 and disc 11 alone are one numerical level, not two anchors.
    supported_levels = (supported_class[:, 1:10] | supported_class[:, 11:20]).sum(1)
    accepted = (proposed != 0) & (support >= minimum_support) & (supported_levels >= minimum_levels)
    offset = torch.where(accepted, proposed, torch.zeros_like(proposed))
    return dict(offset=offset, proposed_offset=proposed, support=support,
                supported_levels=supported_levels, accepted=accepted, scores=scores,
                correspondence_pixels=selected.sum((1, 2)), counts=counts)


@torch.no_grad()
def transport_peer(own: Tensor, peer: Tensor, offset: Tensor, *, confidence=.8):
    """Move peer k+offset to own k, preserving confident absolute agreements.

    Background/canal channels are unchanged. Invalid ordinal mass is dropped,
    not renormalized: an ambiguous 0.4 target must never become confidence 1.0.
    A zero offset returns the exact original tensor values. This function has
    no ground-truth, crop-origin, case-ID, or lowest-visible-ID input.
    """
    own, peer = _pair(own, peer)
    if (offset.shape != (own.shape[0],) or offset.dtype != torch.long
            or offset.device != own.device or (offset.abs() > 8).any()
            or not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise ValueError('Offset must be one integer in [-8,8] per image')
    ids = torch.arange(20, device=own.device)[None].expand(own.shape[0], -1)
    numbered = ((ids >= 1) & (ids <= 9)) | (ids >= 11)
    source = torch.where(numbered, ids + offset[:, None], ids)
    valid = (~numbered | (((ids <= 9) & (source >= 1) & (source <= 9))
                         | ((ids >= 11) & (source >= 11) & (source <= 19))))
    index = source.clamp(0, 19)[:, :, None, None].expand_as(peer)
    moved = peer.gather(1, index) * valid[:, :, None, None]
    oc, oi = own.max(1); pc, pi = peer.max(1)
    agreement = (oi == pi) & (oc >= confidence) & (pc >= confidence)
    return torch.where(agreement[:, None], peer, moved)


@torch.no_grad()
def cross_targets(first: Tensor, second: Tensor, **options):
    """Prepare two directional targets BEFORE any spatial CutMix operation."""
    evidence = estimate_offset(first, second, **options)
    confidence = options.get('confidence', .8)
    to_first = transport_peer(first, second, evidence['offset'], confidence=confidence)
    to_second = transport_peer(second, first, -evidence['offset'], confidence=confidence)
    return to_first, to_second, evidence


@torch.no_grad()
def anchored_mean(own: Tensor, peer: Tensor, transported: Tensor, offset: Tensor):
    """Normalized diagnostic mixture; NOT the actual hard cross-teaching target.

    Incomplete peer support has less mixture weight. Zero offset is exactly the
    original arithmetic mean. Numbering remains anchored to own, not corrected
    against labels. This two-network output is not a single-network result.
    """
    own, peer = _pair(own, peer)
    if transported.shape != own.shape or offset.shape != (own.shape[0],):
        raise ValueError('Mismatched transported target or offset')
    total = own + transported.detach()
    fused = total / total.sum(1, keepdim=True).clamp_min(1e-30)
    return torch.where((offset == 0)[:, None, None, None], (own + peer) / 2, fused)
