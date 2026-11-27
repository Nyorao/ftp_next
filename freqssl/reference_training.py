"""Mean Teacher with independently routed references; no frozen-source gradients."""
import torch

from .augment import aligned_cutmix, strong_view
from .objectives import segmentation_loss, masked_pseudo_loss, linear_ramp
from .reference_conditioning import row_prior


NEIGHBOR_INTENSITY_OFFSETS = (800000011, 1000000007)


def gt_probability(step):
    return .5 * max(0., 1. - step / 3000.)


def soft_cycle_loss(logits, teacher, cfg):
    teacher = teacher.detach()
    confidence, label = teacher.max(1)
    accepted = confidence >= torch.where(label == 0, cfg.background_confidence, cfg.confidence)
    weight = torch.where(label == 0, cfg.background_weight, 1.) * accepted
    return (-(teacher * logits.log_softmax(1)).sum(1) * weight).mean(), float(accepted.float().mean())


def train_reference_step(model, optimizer, batch, cfg, device, step):
    model.train()
    image = batch['image'].to(device)
    label = batch['label'].to(device)
    labeled = batch['is_labeled'].to(device).bool()
    if not labeled.any() or not (~labeled).any():
        raise ValueError('Reference training requires both labeled and unlabeled targets')
    reference = {key: batch['reference_' + key].to(device) for key in ('image', 'probabilities', 'si', 'valid')}
    if cfg.reference_condition == 'row':
        reference['prior'] = batch['reference_prior'].to(device)
    neighbors = getattr(cfg, 'reference_sources', 'case') in ('neighbors', 'case_neighbors')
    if neighbors:
        for key in ('neighbor_image', 'neighbor_probabilities', 'neighbor_valid'):
            reference[key] = batch['reference_' + key].to(device)
    select = lambda values, mask: {k: v[mask] for k, v in values.items()}
    with torch.no_grad():
        target = model.teachers[0](image[~labeled], reference=select(reference, ~labeled)).softmax(1)
    seed = cfg.seed + step * 104729
    strong = strong_view(image, seed, cfg.noise_std, cfg.gain_std)
    condition = dict(reference)
    condition['image'] = strong_view(reference['image'], seed + 600000101, cfg.noise_std, cfg.gain_std)
    if neighbors:
        # One independent intensity stream per slot; offsets do not reuse any
        # existing stream, so the target, case reference and CutMix are unchanged.
        condition['neighbor_image'] = torch.cat([
            strong_view(reference['neighbor_image'][:, j:j+1], seed + offset, cfg.noise_std, cfg.gain_std)
            for j, offset in enumerate(NEIGHBOR_INTENSITY_OFFSETS)], 1)
    selected_gt = torch.zeros(len(image), dtype=torch.bool, device=device)
    if cfg.reference_gt:
        generator = torch.Generator().manual_seed(seed + 91349)
        selected_gt = (torch.rand(len(image), generator=generator).to(device) < gt_probability(step))
        selected_gt &= batch['reference_gt_available'].to(device).bool()
        selected_gt &= ~batch['reference_is_target'].to(device).bool()
        for key, source in (('probabilities', 'gt'), ('prior', 'gt_prior')):
            if key in condition:
                condition[key] = torch.where(selected_gt[:, None, None, None],
                    batch['reference_' + source].to(device), condition[key])
    unmixed = strong
    mixed, target, donor, mask = aligned_cutmix(strong[~labeled], target, seed + 17,
                                               cfg.cutmix_probability, return_routing=True)
    strong = strong.clone(); strong[~labeled] = mixed
    indices = torch.arange(len(image), device=device)
    indices[~labeled] = indices[~labeled][donor]
    condition['donor'] = indices
    condition['mix'] = image.new_zeros((len(image), 1, *image.shape[-2:]))
    condition['mix'][~labeled] = mask
    optimizer.zero_grad(set_to_none=True)
    student = model.students[0]
    logits = student(strong, reference=condition)
    supervised = segmentation_loss(logits[labeled], label[labeled])
    unsupervised, coverage = masked_pseudo_loss(logits[~labeled], target,
        cfg.confidence, cfg.background_confidence, cfg.background_weight)
    ramp = linear_ramp(step, cfg.warmup_steps, cfg.ramp_steps)
    weight = cfg.unsupervised_weight * ramp
    loss = supervised + weight * unsupervised
    cycle = logits.sum() * 0.
    cycle_coverage, cycle_samples = 0., 0
    if cfg.reference_cycle_weight and ramp > 0:
        eligible = (~labeled) & batch['reference_unlabeled_case'].to(device).bool()
        eligible &= ~batch['reference_is_target'].to(device).bool()
        cycle_samples = int(eligible.sum())
        if cycle_samples:
            forward_reference = select({k: v for k, v in condition.items() if k not in ('donor', 'mix')}, eligible)
            forward_prediction = student(unmixed[eligible], reference=forward_reference).softmax(1)
            reverse_reference = dict(image=unmixed[eligible, 1:2], probabilities=forward_prediction,
                si=reference['si'][eligible], valid=reference['valid'][eligible])
            reverse_images = strong_view(batch['reference_triplet'].to(device)[eligible],
                seed + 600000101, cfg.noise_std, cfg.gain_std)
            reconstructed = student(reverse_images, reference=reverse_reference)
            cycle, cycle_coverage = soft_cycle_loss(reconstructed, reference['probabilities'][eligible], cfg)
            loss = loss + cfg.reference_cycle_weight * ramp * cycle
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite reference loss at step {step}')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    metrics = dict(loss=float(loss.detach()), supervised_s1=float(supervised.detach()),
        unsupervised_s1=float(unsupervised.detach()), unsupervised_weight=weight,
        grad_norm_s1=float(norm), ema_decay=model.update_teachers(),
        reference_gt_probability=gt_probability(step) if cfg.reference_gt else 0.,
        reference_gt_used=int(selected_gt.sum()), reference_gt_available=int(batch['reference_gt_available'].sum()),
        reference_self_pairs=int(batch['reference_is_target'].sum()),
        reference_names=list(batch['reference_name']), reference_indices=batch['reference_index'].tolist(),
        cycle_loss=float(cycle.detach()), cycle_coverage=cycle_coverage, cycle_samples=cycle_samples,
        weighted_cycle_loss=cfg.reference_cycle_weight * ramp * float(cycle.detach()),
        context_gamma=[float(m.gamma.detach()) for m in student.context_fusion])
    metrics.update({key + '_s1': value for key, value in coverage.items()})
    if cfg.reference_condition != 'row':
        metrics['reference_gates'] = [float(m.gate.detach().tanh()) for m in student.reference_adapters]
    if neighbors:
        metrics['reference_sources'] = cfg.reference_sources
        metrics['reference_neighbor_gates'] = [float(m.gate.detach().tanh()) for m in student.neighbor_adapters]
        metrics['reference_neighbor_slots'] = int(batch['reference_neighbor_present'].sum())
    return metrics
