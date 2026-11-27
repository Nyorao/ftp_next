"""Image-only canvas interventions with exact transport of their common content."""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ContentPlacement:
    ap_start: int
    si_start: int
    size: int = 160

    def extract(self, tensor: Tensor) -> Tensor:
        if tensor.ndim < 3 or min(tensor.shape[-2:]) < self.size:
            raise ValueError('Expected spatial predictions containing the full placed content')
        if self.ap_start+self.size > tensor.shape[-2] or self.si_start+self.size > tensor.shape[-1]:
            raise ValueError('Content placement falls outside the prediction')
        return tensor[..., self.ap_start:self.ap_start+self.size, self.si_start:self.si_start+self.size]


PLACEMENTS = (ContentPlacement(32, 16), ContentPlacement(32, 48))


def canvas_pair(images: Tensor):
    """Same 160x160 content, shifted 32 pixels along audited cranio-caudal axis.

    Images use the repository's transposed (AP,SI) axes. One shared bilinear
    resize precedes BOTH placements; integer transport then needs no resampling.
    Neither view crops tissue. Both have >=16 pixels of boundary margin. Padding
    is the image's minimum intensity, and no view-specific normalization occurs.
    This is a controlled sensitivity probe, not an inference improvement method.
    """
    if images.ndim != 4 or images.shape[1:] != (1,224,224) or not torch.isfinite(images).all():
        raise ValueError('Expected finite normalized images with shape (N,1,224,224)')
    content = F.interpolate(images, size=(160,160), mode='bilinear', align_corners=False)
    background = images.amin(dim=(-2,-1), keepdim=True)
    views = []
    for position in PLACEMENTS:
        view = background.expand_as(images).clone()
        view[..., position.ap_start:position.ap_start+position.size,
                   position.si_start:position.si_start+position.size] = content
        views.append(view)
    return tuple(views), PLACEMENTS
