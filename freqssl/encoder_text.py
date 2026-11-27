"""PSGM-inspired spatial text fusion inside the shared single-slice encoder.

Token and CLS arms use the same operator and layer-12 bank. Dense anchor
supervision is separate from fusion and identical at E2/E3/E4 in every text arm.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical_text import TOKEN_LENGTH, TEXT_WIDTH
from .models import _group_count
from .slice_context import ContextUNet

STAGES = (2, 3, 4)
ALIGNMENT_TEMPERATURE = 0.07


class EncoderSpatialText(nn.Module):
    def __init__(self, channels, dim=64, bank=None, feature_kind="tokens", active_stage=2):
        super().__init__()
        if feature_kind not in ("tokens", "global_cls", "none"):
            raise ValueError("Unknown encoder text feature kind")
        if ((feature_kind == "none") != (active_stage is None)
                or (active_stage is not None and active_stage not in STAGES)):
            raise ValueError("Select one encoder stage, or none for alignment-only")
        if bank is None:
            bank = (torch.full((12, 4, TOKEN_LENGTH, TEXT_WIDTH), float("nan")),
                    torch.ones(4, TOKEN_LENGTH, dtype=torch.bool),
                    torch.full((4, 512), float("nan")))
        for name, value in zip(("hidden_tokens", "token_mask", "anchors"), bank):
            self.register_buffer(name, value.detach().clone())
        self.feature_kind, self.active_stage, self.dim = feature_kind, active_stage, dim
        self.key = nn.Linear(TEXT_WIDTH, dim, bias=False)
        self.value = nn.Linear(TEXT_WIDTH, dim, bias=False)
        self.norms = nn.ModuleList(nn.LayerNorm(c) for c in channels)
        self.queries = nn.ModuleList(nn.Linear(c, dim, bias=False) for c in channels)
        self.fusions = nn.ModuleList(nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim), nn.GELU(),
            nn.Conv2d(dim, c, 1, bias=False)) for c in channels)
        self.gates = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in STAGES)
        self.alignment_heads = nn.ModuleList(nn.Conv2d(c, 512, 1, bias=False) for c in channels)
        # Initialize every arm identically, then freeze branches that cannot
        # participate. Report effective parameters separately from stored ones.
        self.key.requires_grad_(active_stage is not None)
        self.value.requires_grad_(active_stage is not None)
        for index, stage in enumerate(STAGES):
            for module in (self.norms[index], self.queries[index], self.fusions[index]):
                module.requires_grad_(stage == active_stage)
            self.gates[index].requires_grad_(stage == active_stage)

    def gate_values(self):
        return torch.stack([gate.detach().tanh() for gate in self.gates]).tolist()

    def text_inputs(self):
        if self.feature_kind == "global_cls":
            return self.hidden_tokens[11, :, :1], torch.ones_like(self.token_mask[:, :1])
        if self.feature_kind == "tokens":
            return self.hidden_tokens[11], self.token_mask
        raise ValueError("Alignment-only has no fusion text inputs")

    def alignment_logits(self, features, index):
        projected = F.normalize(self.alignment_heads[index](features), dim=1)
        return torch.einsum("bdhw,cd->bchw", projected, self.anchors) / ALIGNMENT_TEMPERATURE

    def fuse(self, features, index, text, mask, diagnostics=False):
        """The same operator accepts either word tokens or one CLS per class."""
        if not mask.any(-1).all():
            raise ValueError("Every class must have at least one valid token")
        text = F.layer_norm(text.masked_fill(~mask[..., None], 0), (TEXT_WIDTH,))
        q = self.queries[index](self.norms[index](features.flatten(2).transpose(1, 2)))
        scores = torch.einsum("bnd,ctd->bnct", q, self.key(text)) / math.sqrt(self.dim)
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        class_prob = (scores.logsumexp(-1) - mask.sum(-1).float().log()).softmax(-1)
        token_prob = scores.softmax(-1)
        within = torch.einsum("bnct,ctd->bncd", token_prob, self.value(text))
        spatial = (class_prob[..., None] * within).sum(2)
        combined = torch.cat((q, spatial), -1).transpose(1, 2).reshape(
            len(features), 2 * self.dim, *features.shape[-2:])
        residual = self.gates[index].tanh() * self.fusions[index](combined)
        output = features + residual
        stats = {}
        if diagnostics:
            with torch.no_grad():
                token_entropy = -(token_prob * token_prob.clamp_min(1e-12).log()).sum(-1)
                lengths = mask.sum(-1).float().log()
                normalized = torch.where(lengths > 0, token_entropy / lengths.clamp_min(1e-12),
                                         torch.zeros_like(token_entropy))
                stats = dict(
                    residual_rms_ratio=(output.detach() - features.detach()).square().mean((1, 2, 3)).sqrt()
                        / features.detach().square().mean((1, 2, 3)).sqrt().clamp_min(1e-12),
                    class_entropy=-(class_prob * class_prob.clamp_min(1e-12).log()).sum(-1).mean(1),
                    token_entropy=(class_prob * token_entropy).sum(-1).mean(1),
                    token_entropy_normalized=(class_prob * normalized).sum(-1).mean(1))
        return output, stats

    def forward(self, features, stage, diagnostics=False):
        if stage != self.active_stage:
            return features, {}
        text, mask = self.text_inputs()
        return self.fuse(features, STAGES.index(stage), text, mask, diagnostics)


class EncoderTextContextUNet(ContextUNet):
    def __init__(self, num_classes, base_channels, max_channels, mode, text_embeddings=None,
                 text_adapter_dim=64, text_feature_kind="tokens", encoder_text_stage=2):
        super().__init__(num_classes, base_channels, max_channels, mode)
        with torch.random.fork_rng(devices=[]):
            self.encoder_text = EncoderSpatialText([self.widths[s] for s in STAGES],
                text_adapter_dim, text_embeddings, text_feature_kind, encoder_text_stage)

    def _encode_chunk(self, images, collect=False, diagnostics=False):
        if images.ndim != 4 or images.shape[1] != 1 or min(images.shape[-2:]) < 16:
            raise ValueError("Expected single slices with spatial size >=16")
        if collect and len(images) % 3:
            raise ValueError("Alignment requires ordered slice triplets")
        features, skips, logits, stats = images, [], [], {}
        for level, encoder in enumerate(self.encoders):
            if level:
                features = self.pool(features)
            features = encoder(features)
            if level in STAGES:
                # Only center slices enter the auxiliary head, before this
                # stage's injection. The engine then selects labeled centers.
                if collect:
                    logits.append(self.encoder_text.alignment_logits(features[1::3], STAGES.index(level)))
                features, current = self.encoder_text(features, level, diagnostics)
                if current:
                    stats.update({f"E{level}_{name}": value[1::3] for name, value in current.items()})
            skips.append(features)
        return skips, dict(tissue_logits=logits, alignment_stages=["E2", "E3", "E4"],
                           encoder_text_diagnostics=stats)

    def encode(self, images):
        if self.training:
            return self._encode_chunk(images)[0]
        chunks = []
        for start in range(0, len(images), self.inference_microbatch):
            values = images[start:start + self.inference_microbatch]
            count = len(values)
            if count < self.inference_microbatch:
                values = torch.cat((values, values.new_zeros(
                    (self.inference_microbatch - count, *values.shape[1:]))))
            chunks.append([value[:count] for value in self._encode_chunk(values)[0]])
        return [torch.cat([chunk[level] for chunk in chunks]) for level in range(5)]

    def forward_with_alignment(self, images, collect_diagnostics=False):
        if not self.training or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Alignment collection requires training image triplets")
        encoded, aux = self._encode_chunk(images.reshape(-1, 1, *images.shape[-2:]),
                                          collect=True, diagnostics=collect_diagnostics)
        triplets = [value.reshape(len(images), 3, *value.shape[1:]) for value in encoded]
        return self.project(self.context_features(triplets)), aux
