"""Training for ordinal ambiguity routed, heterogeneous dual students."""
import torch

from .anatomy import (family_labels, family_log_probabilities, marginal_set_loss,
                      route_targets)
from .augment import aligned_cutmix, strong_view
from .objectives import segmentation_loss, linear_ramp, masked_pseudo_loss


def train_ordinal_step(model, optimizer, batch, cfg, device, step):
    model.train()
    images = batch['image'].to(device=device, dtype=torch.float32)
    labels = batch['label'].to(device=device, dtype=torch.long)
    labeled = batch['is_labeled'].to(device=device, dtype=torch.bool)
    if not labeled.any() or not (~labeled).any():
        raise ValueError('Both labeled and unlabeled samples are required')
    weight = cfg.unsupervised_weight * linear_ramp(step, cfg.warmup_steps, cfg.ramp_steps)
    with torch.no_grad():
        probabilities = [model.teacher_logits(images[~labeled], i).float().softmax(1) for i in range(2)]
    optimizer.zero_grad(set_to_none=True)
    metrics = {'unsupervised_weight': weight, 'loss': 0.0}
    # The allowed sets are symmetric under exchanging teachers. Compute the
    # consensus once, then apply each student's own spatial CutMix to a copy.
    if cfg.ordinal_use_routing:
        packed, routing = route_targets(probabilities[1], probabilities[0], confidence=cfg.confidence,
            background_confidence=cfg.background_confidence, background_weight=cfg.background_weight,
            chain_threshold=cfg.ordinal_chain_threshold, interval_mass=cfg.ordinal_interval_mass,
            max_interval=cfg.ordinal_max_interval, bins=cfg.ordinal_bins, use_chain=cfg.ordinal_use_chain)
    for i in range(2):
        seed = cfg.seed + step * 104729 + i * 10000019
        strong = strong_view(images, seed, cfg.noise_std, cfg.gain_std)
        # Structural evidence was computed on whole weak images. Splice only
        # pixelwise target sets/weights; never derive a chain across CutMix seams.
        mixed, target = aligned_cutmix(strong[~labeled], packed if cfg.ordinal_use_routing else probabilities[1-i],
                                      seed+17, cfg.cutmix_probability)
        strong = strong.clone()
        strong[~labeled] = mixed
        logits = model.student_logits(strong, i).float()
        supervised = segmentation_loss(logits[labeled], labels[labeled])
        tissue = segmentation_loss(family_log_probabilities(logits[labeled]), family_labels(labels[labeled]))
        if cfg.ordinal_use_routing:
            unsupervised = marginal_set_loss(logits[~labeled], target)
        else:
            unsupervised, routing = masked_pseudo_loss(logits[~labeled], target, cfg.confidence,
                                                      cfg.background_confidence, cfg.background_weight)
        loss = supervised + cfg.ordinal_tissue_weight * tissue + weight * unsupervised
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite ordinal loss at step {step}, student {i+1}')
        loss.backward()
        metrics.update({f'{key}_s{i+1}': value for key, value in routing.items()})
        metrics.update({f'supervised_s{i+1}':float(supervised.detach()),
                        f'tissue_supervised_s{i+1}':float(tissue.detach()),
                        f'unsupervised_s{i+1}':float(unsupervised.detach()),
                        f'weighted_pseudo_ce_s{i+1}':weight*float(unsupervised.detach()),
                        f'target_source_s{i+1}': ('joint_teacher_allowed_sets' if cfg.ordinal_use_routing else f'teacher{2-i}'),
                        f'pseudo_dice_s{i+1}':0., f'region_s{i+1}':0.,
                        f'weighted_pseudo_dice_s{i+1}':0.,f'weighted_region_s{i+1}':0.})
        metrics['loss'] += float(loss.detach())
    for i, student in enumerate(model.students):
        metrics[f'grad_norm_s{i+1}'] = float(torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip, error_if_nonfinite=True))
    optimizer.step()
    metrics['ema_decay'] = model.update_teachers()
    first, second = [p.argmax(1) for p in probabilities]
    fg = (first != 0) | (second != 0)
    metrics['teacher_disagreement'] = float((first != second).float().mean())
    metrics['foreground_disagreement'] = float(((first != second) & fg).sum()/fg.sum().clamp_min(1))
    return metrics
