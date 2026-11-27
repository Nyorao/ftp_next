"""Read-only interventions on a frozen three-slice segmentation network.

These are model-dependence probes, not causal estimates of anatomical utility.
Ground truth is used only in scoring/stratification, never in input selection.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, laplace, median_filter
import torch
from torch.nn import functional as F

from .slice_context import ContextUNet


CONDITIONS = ("full", "no_previous", "no_next", "center_only", "swapped", "gap2", "gap3")
REGIONS = ("all", "foreground", "background", "vertebra", "canal", "disc", "boundary", "interior")
FAMILY = np.array([0] + [1] * 9 + [2] + [3] * 9, dtype=np.uint8)


def context_indices(index, count, condition):
    """Return acquisition-order indices; clipping never crosses patients."""
    if count < 1 or not 0 <= index < count or condition not in CONDITIONS:
        raise ValueError("Invalid center, count, or intervention")
    radius = {"gap2": 2, "gap3": 3}.get(condition, 1)
    offsets = (-radius, 0, radius)
    if condition == "no_previous":
        offsets = (0, 0, 1)
    elif condition == "no_next":
        offsets = (-1, 0, 0)
    elif condition == "center_only":
        offsets = (0, 0, 0)
    elif condition == "swapped":
        offsets = (1, 0, -1)
    return tuple(min(count - 1, max(0, index + delta)) for delta in offsets)


def slice_metadata(index, count, spacing_mm, reverse_physical_order):
    if count < 1 or not 0 <= index < count or not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Invalid slice geometry")
    u = index / (count - 1) if count > 1 else .5
    physical_u = 1 - u if reverse_physical_order else u
    sign = -1 if reverse_physical_order else 1
    return dict(index=index, count=count, position=u, position_bin=min(4, int(u * 5)),
                physical_position=physical_u, physical_position_bin=min(4, int(physical_u * 5)),
                spacing_mm=float(spacing_mm), spacing_bin="thin" if spacing_mm < 3.8 else
                ("medium" if spacing_mm < 4.6 else "thick"),
                increasing_index_physical_direction="right" if sign < 0 else "left",
                previous_physical_direction="left" if sign < 0 else "right",
                next_physical_direction="right" if sign < 0 else "left",
                endpoint=index in (0, count - 1), both_adjacent_real=0 < index < count - 1,
                both_gap2_real=2 <= index < count - 2, both_gap3_real=3 <= index < count - 3,
                signed_mm_per_index=float(sign * spacing_mm))


def region_masks(target, boundary_radius=2):
    """Two-model-pixel band around in-plane class transitions; not a mm metric."""
    target = np.asarray(target)
    if target.ndim != 2 or not np.isin(target, np.arange(20)).all():
        raise ValueError("Expected a 2D class mask in 0..19")
    edge = np.zeros_like(target, dtype=bool)
    for axis in (0, 1):
        diff = np.diff(target, axis=axis) != 0
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        edge[tuple(a)] |= diff
        edge[tuple(b)] |= diff
    band = binary_dilation(edge, iterations=boundary_radius) if boundary_radius else edge
    family = FAMILY[target]
    return dict(all=np.ones_like(edge), foreground=target > 0, background=target == 0,
                vertebra=family == 1, canal=family == 2, disc=family == 3,
                boundary=band, interior=(target > 0) & ~band)


def image_quality(image):
    """Image-only texture proxies; neither is a calibrated quality score."""
    image = np.asarray(image, dtype=np.float32)
    residual = image - median_filter(image, size=3)
    return dict(laplacian_variance=float(laplace(image).var()),
                high_frequency_mad=float(np.median(np.abs(residual - np.median(residual)))))


def feature_relationship(center, neighbor, radius=1):
    """Local cosine matching diagnostic, not registration or anatomical truth."""
    if center.shape != neighbor.shape or center.ndim != 3:
        raise ValueError("Expected equal C,H,W feature maps")
    a = F.normalize(center.float(), dim=0, eps=1e-8)
    b = F.normalize(neighbor.float(), dim=0, eps=1e-8)
    same = (a * b).sum(0)
    best = same.clone()
    shifted = torch.zeros_like(same)
    height, width = same.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not (dy or dx):
                continue
            ya, yb = max(0, -dy), min(height, height - dy)
            xa, xb = max(0, -dx), min(width, width - dx)
            score = (a[:, ya:yb, xa:xb] * b[:, ya+dy:yb+dy, xa+dx:xb+dx]).sum(0)
            improved = score > best[ya:yb, xa:xb]
            best[ya:yb, xa:xb] = torch.maximum(score, best[ya:yb, xa:xb])
            shifted[ya:yb, xa:xb][improved] = float((dy * dy + dx * dx) ** .5)
    return {"cosine_distance": 1 - same, "local_match_gain": best - same,
            "best_shift_feature_pixels": shifted}


@torch.inference_mode()
def encode_volume(backbone, images):
    if not isinstance(backbone, ContextUNet) or backbone.training:
        raise ValueError("An eval ContextUNet is required")
    if images.ndim != 4 or images.shape[1] != 1 or not len(images):
        raise ValueError("Expected nonempty N,1,H,W images")
    return backbone.encode(images)


@torch.inference_mode()
def predict_encoded(backbone, encoded, condition, batch_size=8, observe=False):
    """Yield unmodified decoder logits plus full-input fusion observations.

    Observer hooks never replace a layer's output. Padding and convolution
    batch shapes remain controlled by ContextUNet itself.
    """
    if not isinstance(backbone, ContextUNet) or backbone.training or batch_size < 1:
        raise ValueError("An eval backbone and positive batch size are required")
    count = len(encoded[0])
    if len(encoded) != 5 or any(len(x) != count for x in encoded):
        raise ValueError("Five consistent encoder levels are required")
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        indices = [context_indices(i, count, condition) for i in range(start, end)]
        triplets = [level[torch.tensor(indices, device=level.device)] for level in encoded]
        captured = [[] for _ in backbone.context_fusion]
        handles = []
        if observe:
            for level, module in enumerate(backbone.context_fusion):
                def hook(layer, inputs, output, level=level):
                    values = inputs[0]
                    weights = (layer.weights(values.flatten(1, 2)).softmax(1) if layer.weights is not None
                               else values.new_full((len(values), 3, *values.shape[-2:]), 1 / 3))
                    center = values[:, 1]
                    terms = layer.gamma * weights[:, (0, 2), None] * (values[:, (0, 2)] - center[:, None])
                    norms = terms.square().mean(2).sqrt()
                    captured[level].append((weights.detach().cpu(), norms.detach().cpu()))
                handles.append(module.register_forward_hook(hook))
        try:
            logits = backbone.project(backbone.context_features(triplets))
        finally:
            for handle in handles:
                handle.remove()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits")
        observations = []
        if observe:
            # Remove each internal microbatch's zero padding, including a final short batch.
            for chunks in captured:
                weights, norms, remaining = [], [], end - start
                for weight, norm in chunks:
                    take = min(remaining, len(weight))
                    weights.append(weight[:take]); norms.append(norm[:take]); remaining -= take
                if remaining:
                    raise AssertionError("Observer missed a fusion batch")
                observations.append((torch.cat(weights), torch.cat(norms)))
        yield start, logits, observations


def score_logits(logits, target, full_prediction, volume_present):
    """224-grid diagnostics; all primary outcomes are separately computed in 3D."""
    target = np.asarray(target, dtype=np.int64)
    logp = logits.float().log_softmax(0)
    prediction = logp.argmax(0).cpu().numpy()
    gt = torch.from_numpy(target).to(logp.device)
    loss = -logp.gather(0, gt[None])[0].cpu().numpy()
    entropy = -(logp.exp() * logp).sum(0).cpu().numpy()
    masks = region_masks(target)
    rows = {}
    absent = (prediction > 0) & ~np.isin(prediction, volume_present)
    slice_absent = (prediction > 0) & ~np.isin(prediction, np.unique(target))
    for region, mask in masks.items():
        n = int(mask.sum())
        classes = np.unique(target[mask])
        foreground_classes = classes[classes != 0]
        rows[region] = dict(pixels=n,
            errors=int(((prediction != target) & mask).sum()),
            changed=int(((prediction != full_prediction) & mask).sum()),
            harmed=int(((full_prediction == target) & (prediction != target) & mask).sum()),
            helped=int(((full_prediction != target) & (prediction == target) & mask).sum()),
            absent_class_pixels=int((absent & mask).sum()),
            slice_absent_class_pixels=int((slice_absent & mask).sum()),
            ce_sum=float(loss[mask].sum(dtype=np.float64)),
            entropy_sum=float(entropy[mask].sum(dtype=np.float64)),
            class_balanced_ce=float(np.mean([loss[mask & (target == c)].mean() for c in classes])) if n else None,
            foreground_class_balanced_ce=float(np.mean([loss[mask & (target == c)].mean()
                for c in foreground_classes])) if len(foreground_classes) else None)
    return prediction, rows


def observed_regions(observations, row, target, gamma):
    """Weights and signed residual scales, averaged in labeled scoring regions."""
    masks = region_masks(target)
    result = {}
    for name, mask in masks.items():
        result[name] = []
        for (weights, norms), scalar in zip(observations, gamma):
            # Pool region occupancy instead of nearest sampling small structures away.
            occupancy = F.adaptive_avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None],
                                             weights.shape[-2:])[0, 0]
            denominator = float(occupancy.sum())
            if not denominator:
                result[name].append(None)
                continue
            w = ((weights[row] * occupancy).sum((1, 2)) / denominator).tolist()
            residual = ((norms[row] * occupancy).sum((1, 2)) / denominator).tolist()
            result[name].append(dict(weights=w, gamma=float(scalar),
                effective_coefficients=[float(scalar*w[0]), float(1-scalar+scalar*w[1]), float(scalar*w[2])],
                neighbor_residual_rms=residual))
    return result


def bootstrap_cases(values, samples=10000, seed=20261105):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return dict(n_cases=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(samples, len(values)))].mean(1)
    return dict(n_cases=len(values), mean=float(values.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist(),
                positive_cases=int((values > 0).sum()), negative_cases=int((values < 0).sum()))


def utility_contrasts(errors):
    """Positive utility means lower error when the specified context is present."""
    full, no_prev, no_next, center = (errors[k] for k in CONDITIONS[:4])
    return dict(previous_given_next=no_prev-full, next_given_previous=no_next-full,
                previous_alone=center-no_next, next_alone=center-no_prev,
                joint=center-full, interaction=no_prev+no_next-full-center,
                previous_minus_next=no_prev-no_next)
