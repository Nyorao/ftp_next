"""CFF-inspired center-residual fusion, not an official CFF reproduction.

Three ordered sagittal slices share the unmodified 2D encoder. Only levels
2/3/4 (stride 4/8/16) fuse; high-resolution skips belong to the center.
"""

import torch
from torch import nn

from .models import UNet


class CenterResidualFusion(nn.Module):
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in ("mean", "adaptive"):
            raise ValueError("Fusion mode must be mean or adaptive")
        self.gamma = nn.Parameter(torch.zeros(()))
        self.weights = nn.Conv2d(3 * channels, 3, 1) if mode == "adaptive" else None
        if self.weights is not None:
            nn.init.zeros_(self.weights.weight)
            nn.init.zeros_(self.weights.bias)

    def forward(self, features):
        if features.ndim != 5 or features.shape[1] != 3:
            raise ValueError("Expected (B,3,C,H,W) features")
        if self.weights is None:
            fused = features.mean(1)
        else:
            weights = self.weights(features.flatten(1, 2)).softmax(1)
            fused = (weights[:, :, None] * features).sum(1)
        center = features[:, 1]
        return center + self.gamma * (fused - center)


class ContextUNet(UNet):
    inference_microbatch = 8

    def __init__(self, num_classes, base_channels, max_channels, mode):
        super().__init__(num_classes, base_channels, max_channels)
        self.slice_context = mode
        # Preserve both the backbone initialization and the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.context_fusion = nn.ModuleList(
                CenterResidualFusion(self.widths[level], mode) for level in (2, 3, 4))

    def encode(self, images):
        if self.training:
            return super().encode(images)
        # cuDNN can select different convolution arithmetic for batches of
        # 1/8/24, including TF32. A fixed eval microbatch makes cached and
        # uncached encodings of a real slice identical. Padding has no effect
        # on other examples because this backbone uses per-example GroupNorm.
        chunks = []
        for start in range(0, len(images), self.inference_microbatch):
            values = images[start:start + self.inference_microbatch]
            count = len(values)
            if count < self.inference_microbatch:
                values = torch.cat((values, values.new_zeros(
                    (self.inference_microbatch-count, *values.shape[1:]))))
            chunks.append([value[:count] for value in super().encode(values)])
        return [torch.cat([chunk[level] for chunk in chunks]) for level in range(5)]

    def context_features(self, triplets):
        if not self.training:
            # Fix fusion/decoder arithmetic too, including one-slice volumes
            # and callers using a different output batch size.
            chunks = []
            for start in range(0, len(triplets[0]), self.inference_microbatch):
                values = [value[start:start + self.inference_microbatch] for value in triplets]
                count = len(values[0])
                if count < self.inference_microbatch:
                    values = [torch.cat((value, value.new_zeros(
                        (self.inference_microbatch-count, *value.shape[1:])))) for value in values]
                chunks.append(self._decode_triplets(values)[:count])
            return torch.cat(chunks)
        return self._decode_triplets(triplets)

    def _decode_triplets(self, triplets):
        skips = [triplets[0][:, 1], triplets[1][:, 1]]
        skips.extend(fuse(value) for fuse, value in zip(self.context_fusion, triplets[2:]))
        return self.decode_features(skips)

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("ContextUNet inputs must have shape (B,3,H,W)")
        encoded = self.encode(images.reshape(-1, 1, *images.shape[-2:]))
        return self.context_features([x.reshape(len(images), 3, *x.shape[1:]) for x in encoded])

    def project(self, features):
        if self.training:
            return self.head(features)
        outputs = []
        for start in range(0, len(features), self.inference_microbatch):
            values = features[start:start + self.inference_microbatch]
            count = len(values)
            if count < self.inference_microbatch:
                values = torch.cat((values, values.new_zeros(
                    (self.inference_microbatch-count, *values.shape[1:]))))
            outputs.append(self.head(values)[:count])
        return torch.cat(outputs)

    def forward(self, images):
        return self.project(self.forward_features(images))

    def forward_with_features(self, images):
        features = self.forward_features(images)
        return self.project(features), features


@torch.inference_mode()
def iter_cached_logits(backbone, count, read_slice, device, batch_size=8):
    """Yield (start, logits), encoding every slice once; cache one moving window.

    read_slice reads an image only and returns a normalized (1,H,W) tensor.
    The cache is local to one volume/network/call and cannot cross patients.
    """
    if not isinstance(backbone, ContextUNet) or backbone.training:
        raise ValueError("Cached context inference requires an eval ContextUNet")
    if count < 1 or batch_size < 1:
        raise ValueError("A nonempty volume and positive batch_size are required")
    cache = {}
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        needed = range(max(0, start - 1), min(count, end + 1))
        missing = [index for index in needed if index not in cache]
        for offset in range(0, len(missing), batch_size):
            indices = missing[offset:offset + batch_size]
            images = torch.stack([read_slice(index) for index in indices]).to(device)
            encoded = backbone.encode(images)
            for row, index in enumerate(indices):
                cache[index] = [value[row] for value in encoded]
        triplets = [torch.stack([
            torch.stack([cache[min(count - 1, max(0, index + delta))][level]
                         for delta in (-1, 0, 1)])
            for index in range(start, end)]) for level in range(5)]
        logits = backbone.project(backbone.context_features(triplets))
        yield start, logits
        cache = {index: values for index, values in cache.items() if index >= end - 1}
