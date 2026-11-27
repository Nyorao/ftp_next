"""Volume-only cache optimization; learned layers and checkpoint keys are unchanged."""

import torch

from .slice_context import ContextUNet


@torch.inference_mode()
def iter_cached_logits(backbone, count, read_slice, device, batch_size=8, *,
                       read_position=None, reference_policy='middle', stats=None):
    """Encode aligned microbatches once and gather high-resolution centers only.

    Prefetching finishes a full encoder microbatch whenever possible. The
    reference pre-encode, its containing block and the tail may need padding.
    GroupNorm is per example; the original
    fixed encoder/fusion/decoder batch shapes and slice order are preserved.
    The cache is bounded by output batch size plus a small prefetch window.
    """
    if not isinstance(backbone, ContextUNet) or backbone.training:
        raise ValueError("Cached context inference requires an eval ContextUNet")
    if count < 1 or batch_size < 1:
        raise ValueError("A nonempty volume and positive batch_size are required")
    cache, encoded_until = {}, 0
    microbatch = backbone.inference_microbatch
    slices = getattr(backbone,'context_slices',3)
    use_position = getattr(backbone,'context_position','none') == 'relative'
    if use_position and read_position is None:
        raise ValueError('Position-aware cache requires image-derived coordinates')
    if not use_position and read_position is not None:
        raise ValueError('Unexpected position inputs')
    if slices != 4 and reference_policy != 'middle':
        raise ValueError('Reference intervention requires four-slice model')
    from .case_context import reference_index
    fixed_reference = reference_index(count,reference_policy) if slices==4 and reference_policy!='target' else None
    if stats is not None:
        stats.update(real_slices_encoded=0,encoder_slots=0,output_slots=0,reference_index=fixed_reference,
                     reference_preencoded_slices=int(fixed_reference is not None),
                     reference_preencode_slots=microbatch if fixed_reference is not None else 0)
    def encode_indices(indices):
        if not indices:
            return
        images = torch.stack([read_slice(i) for i in indices]).to(device)
        encoded = backbone.encode(images)
        for row,index in enumerate(indices):
            cache[index] = [v[row] for v in encoded]
        if stats is not None:
            stats['real_slices_encoded'] += len(indices)
            stats['encoder_slots'] += microbatch*((len(indices)+microbatch-1)//microbatch)
    if fixed_reference is not None:
        encode_indices([fixed_reference])
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        needed_until = min(count, end + 1)
        while encoded_until < needed_until:
            stop = min(count, encoded_until + microbatch)
            encode_indices([i for i in range(encoded_until,stop) if i not in cache])
            encoded_until = stop
        triplets = []
        for level in range(5):
            if level < 2:
                # _decode_triplets uses only [:, 1] at these two levels. An
                # expanded view avoids materializing two unused neighbor maps.
                centers = torch.stack([cache[i][level] for i in range(start, end)])
                triplets.append(centers[:, None].expand(-1, slices, -1, -1, -1))
            else:
                triplets.append(torch.stack([torch.stack([
                    cache[j][level] for j in
                    ([min(count-1,max(0,i+d)) for d in (-1,0,1)] +
                     ([reference_index(count,reference_policy,i)] if slices==4 else []))])
                    for i in range(start, end)]))
        position = (triplets[0].new_tensor([read_position(i) for i in range(start,end)]).reshape(-1,1,1,1)
                    if use_position else None)
        features = backbone.context_features(triplets,**({} if position is None else {'position':position}))
        logits = backbone.project(features)
        if stats is not None:
            stats['output_slots'] += microbatch*((end-start+microbatch-1)//microbatch)
            stats['encoder_padding_slots'] = stats['encoder_slots']-stats['real_slices_encoded']
            stats['output_padding_slots'] = stats['output_slots']-end
        yield start, logits
        cache = {i:values for i,values in cache.items() if i >= end-1 or i == fixed_reference}
