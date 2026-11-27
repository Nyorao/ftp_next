"""Tissue/relative-level factorization with an axial context head."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import UNet, _group_count


class OrdinalUNet(UNet):
    """Same spatial backbone, different label factorization from the flat student.

    Four tissue logits and two nine-way level conditionals form a normalized
    20-class distribution. A pooled axial context path adds whole-slice evidence
    to local level logits. Shape can learn without choosing an ambiguous level.
    There is no hard sequence constraint at inference and no extra annotation.
    """
    def __init__(self, num_classes=20, base_channels=32, max_channels=256):
        if num_classes != 20:
            raise ValueError("OrdinalUNet requires SPIDER's 20-label ontology")
        super().__init__(num_classes, base_channels, max_channels)
        self.head = nn.Conv2d(base_channels, 22, 1)
        layers = []
        for dilation in (1, 2, 4, 8):
            layers.extend((nn.Conv1d(base_channels, base_channels, 5, padding=2*dilation,
                                     dilation=dilation, bias=False),
                           nn.GroupNorm(_group_count(base_channels), base_channels),
                           nn.LeakyReLU(0.01)))
        layers.append(nn.Conv1d(base_channels, 18, 1))
        self.axial = nn.Sequential(*layers)
        # Initialize new heads without reinitializing the shared backbone.
        self._initialize(self.head)
        for module in self.axial.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.axial[-1].weight)
        nn.init.zeros_(self.axial[-1].bias)

    def _factorized_logits(self, features):
        local = self.head(features).float()
        tissue = local[:, :4].log_softmax(1)
        # Work at <=28 positions so the dilated context covers the complete axis.
        width = features.shape[-1]
        bins = max(size for size in range(1, min(28, width) + 1) if width % size == 0)
        block = width // bins
        pooled = features.reshape(*features.shape[:3], bins, block).mean((2, 4))
        # Block expansion has a deterministic backward on torch 2.0 CUDA;
        # linear1d interpolation and adaptive pooling backwards do not.
        context = self.axial(pooled).repeat_interleave(block, dim=-1)
        levels = local[:, 4:] + context[:, :, None]
        vertebra = tissue[:, 1:2] + levels[:, :9].log_softmax(1)
        disc = tissue[:, 3:4] + levels[:, 9:].log_softmax(1)
        return torch.cat((tissue[:, :1], vertebra, tissue[:, 2:3], disc), 1)

    def forward(self, images):
        return self._factorized_logits(self.forward_features(images))

    def forward_with_features(self, images):
        features = self.forward_features(images)
        return self._factorized_logits(features), features
