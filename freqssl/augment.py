"""Independent intensity perturbations and spatially matched pseudo-label CutMix."""

import torch


def strong_view(images, seed, noise_std, gain_std):
    if images.shape[1] in (3,4):
        # The center sees exactly the old random stream, even with neighbors.
        return torch.cat([strong_view(images[:, i:i+1], seed + offset, noise_std, gain_std)
                          for i, offset in enumerate((200000033, 0, 400000067,600000101)[:images.shape[1]])], dim=1)
    generator = torch.Generator(device=images.device).manual_seed(seed)
    gain = torch.randn((len(images), 1, 1, 1), generator=generator,
                       device=images.device).mul(gain_std).clamp(-0.3, 0.3).add(1)
    noise = torch.randn(images.shape, generator=generator, device=images.device)
    return images * gain + noise.clamp(-2, 2) * noise_std


def aligned_cutmix(images, probabilities, seed, probability, *, return_routing=False):
    """Mix only an unlabeled minibatch; use exactly the same donors and rectangle.

    CPU RNG keeps augmentation choices independent of model count and device.
    Returns fresh tensors, so neither weak teacher inputs nor labels are mutated.
    """
    if len(images) != len(probabilities) or images.shape[-2:] != probabilities.shape[-2:]:
        raise ValueError("CutMix image and target coordinates must match")
    generator = torch.Generator().manual_seed(seed)
    identity = torch.arange(len(images), device=images.device)
    mask = images.new_zeros((len(images), 1, *images.shape[-2:]))
    if len(images) < 2 or torch.rand((), generator=generator).item() >= probability:
        result = (images, probabilities.detach())
        return (*result, identity, mask) if return_routing else result
    height, width = images.shape[-2:]
    h = max(1, int(height * (0.3 + 0.4 * torch.rand((), generator=generator).item())))
    w = max(1, int(width * (0.3 + 0.4 * torch.rand((), generator=generator).item())))
    y = int(torch.randint(height - h + 1, (), generator=generator))
    x = int(torch.randint(width - w + 1, (), generator=generator))
    # A cyclic nonzero shift guarantees a different donor for every sample.
    shift = int(torch.randint(1, len(images), (), generator=generator))
    donor = torch.arange(len(images), device=images.device).roll(shift)
    mixed, target = images.clone(), probabilities.detach().clone()
    mixed[:, :, y:y+h, x:x+w] = images[donor, :, y:y+h, x:x+w]
    target[:, :, y:y+h, x:x+w] = probabilities.detach()[donor, :, y:y+h, x:x+w]
    mask[:, :, y:y+h, x:x+w] = 1
    return (mixed, target, donor, mask) if return_routing else (mixed, target)
