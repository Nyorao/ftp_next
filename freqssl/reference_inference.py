"""Bounded per-volume target feature cache and a single immutable reference bank."""
import time

import torch

from .reference_conditioning import si_coordinates, row_prior
from .reference_data import canonical_probabilities, probability_statistics, select_reference, file_hash


@torch.inference_mode()
def iter_reference_logits(backbone, count, read_image, read_probability, device, batch_size=8,
                          *, reference_index=None, reference_image=None, reference_probability=None,
                          disable=False, row_shift=False, stats=None):
    if backbone.training or count < 1 or batch_size < 1:
        raise ValueError('Reference cache requires eval mode and a nonempty case')
    if reference_index is None:
        reference_index = select_reference([probability_statistics(read_probability(i)) for i in range(count)])
    if not 0 <= reference_index < count:
        raise ValueError('Invalid reference index')
    cache = {}
    micro = backbone.inference_microbatch
    counters = dict(real_slices_encoded=0, encoder_slots=0, output_slots=0, reference_index=reference_index)
    def encode(indices):
        images = torch.stack([read_image(i) for i in indices]).to(device)
        values = backbone.encode(images)
        counters['real_slices_encoded'] += len(indices)
        counters['encoder_slots'] += micro * ((len(indices) + micro - 1) // micro)
        for j, index in enumerate(indices):
            cache[index] = [v[j] for v in values]
    bank = None
    if not disable:
        image = (read_image(reference_index) if reference_image is None else reference_image)[None].to(device)
        probability = (read_probability(reference_index) if reference_probability is None else reference_probability)[None].to(device)
        si = si_coordinates(1, image.shape[-2:], device)
        valid = torch.ones_like(si)
        encoded = None
        if backbone.reference_condition != 'row' and reference_image is None:
            encode([reference_index])
            encoded = [v[None] for v in cache[reference_index]]
        prior = row_prior(probability)
        if row_shift:
            prior = prior.roll(prior.shape[-1] // 4, -1)
            probability = probability.roll(probability.shape[-1] // 4, -1)
        bank = backbone.prepare_reference(image, probability, si, valid, prior, encoded)
    encoded_until = 0
    # Fixed output microbatches preserve arithmetic independently of caller batching.
    for start in range(0, count, micro):
        end = min(count, start + micro)
        while encoded_until < min(count, end + 1):
            stop = min(count, encoded_until + micro)
            missing = [i for i in range(encoded_until, stop) if i not in cache]
            if missing:
                encode(missing)
            encoded_until = stop
        triplets = []
        for level in range(5):
            if level < 2:
                centers = torch.stack([cache[i][level] for i in range(start, end)])
                value = centers[:, None].expand(-1, 3, -1, -1, -1)
            else:
                value = torch.stack([torch.stack([cache[min(count-1, max(0, i+d))][level]
                                                  for d in (-1, 0, 1)]) for i in range(start, end)])
            triplets.append(backbone._pad(value))
        shape = triplets[0].shape[-2:]
        coords = si_coordinates(micro, shape, device)
        valid = torch.ones_like(coords); valid[end-start:] = 0
        local_bank = (None if bank is None else backbone.select_bank(bank, torch.zeros(micro, device=device, dtype=torch.long)))
        features = backbone.decode_reference(triplets, local_bank, coords, valid)
        logits = backbone.project(features)[:end-start]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite reference inference')
        counters['output_slots'] += micro
        yield start, logits
        cache = {i: value for i, value in cache.items() if i >= end-1 or (bank is not None and i == reference_index)}
    if stats is not None:
        stats.update(counters)


@torch.inference_mode()
def frozen_source_predictions(checkpoint, count, read_image, device, expected_hash='', batch_size=8):
    from .inference import load_single_network
    from .context_cache import iter_cached_logits
    if expected_hash and file_hash(checkpoint) != expected_hash:
        raise ValueError('Frozen reference source changed')
    started = time.monotonic()
    source, cfg, receipt = load_single_network(checkpoint, device, 'teacher1')
    if cfg.reference_condition != 'none' or cfg.case_reference != 'none' or cfg.context_position != 'none' or cfg.slice_context != 'adaptive':
        raise ValueError('Reference source must be the unconditioned A network')
    from experiments.profile_context_efficiency import ConvCounter
    counter = ConvCounter(source.backbone)
    probabilities = []
    try:
        for _, logits in iter_cached_logits(source.backbone, count, read_image, device, batch_size):
            # Match the cached FP16 roundtrip and per-slice CPU reduction exactly.
            stored = logits.float().softmax(1).half().cpu()
            probabilities.extend(canonical_probabilities(value) for value in stored)
    finally:
        counter.close()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    receipt.update(seconds=time.monotonic()-started, conv_macs=sum(counter.macs.values()),
                   checkpoint_sha256=file_hash(checkpoint), annotation_files_read=0)
    return probabilities, receipt
