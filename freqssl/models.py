"""Independent U-Nets and paired EMA teachers for frequency-view experiments.

The two-view design is inspired by frequency-aware segmentation and cross-pseudo
supervision. It is a deliberately small experimental baseline, not an official
reimplementation of XNet, FRCNet, or another published architecture.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


METHODS = frozenset(
    {"supervised", "mean_teacher", "dual_independent", "dual_cps", "dual_frequency", "dual_ordinal"}
)


def haar_components(images: Tensor) -> tuple[Tensor, Tensor]:
    """Return an LL reconstruction and its signed high-frequency residual.

    Averaging non-overlapping 2 x 2 blocks and repeating each average reconstructs
    the low-frequency part of a one-level orthonormal Haar transform. The
    residual contains all three detail bands in image coordinates, so ``low +
    high`` reconstructs the input. Replicate padding handles odd image sizes;
    both outputs are cropped back to the original size. Half precision and
    integer inputs are promoted to float32 without leaving the input device.
    """
    if images.ndim != 4 or images.shape[-2] == 0 or images.shape[-1] == 0:
        raise ValueError("Haar inputs must have shape (N, C, H, W) with H, W > 0")
    values = images if images.dtype == torch.float64 else images.float()
    height, width = values.shape[-2:]
    padded = F.pad(values, (0, width % 2, 0, height % 2), mode="replicate")
    averages = F.avg_pool2d(padded, kernel_size=2, stride=2)
    low = F.interpolate(averages, scale_factor=2, mode="nearest")
    low = low[..., :height, :width]
    return low, values - low


def _group_count(channels: int) -> int:
    # At least two channels per group also permits batch size 1 at a 1 x 1
    # bottleneck, unlike GroupNorm with one channel per group.
    for groups in range(min(8, channels // 2), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = _group_count(out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, skip_channels, 3, padding=1, bias=False)
        self.fuse = _ConvBlock(2 * skip_channels, skip_channels)

    def forward(self, features: Tensor, skip: Tensor) -> Tensor:
        features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
        features = self.project(features)
        return self.fuse(torch.cat((skip, features), dim=1))


class UNet(nn.Module):
    """Single-channel, four-downsampling U-Net with no pretrained components."""

    def __init__(self, num_classes: int, base_channels: int, max_channels: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        if base_channels < 2 or max_channels < 2:
            raise ValueError("base_channels and max_channels must be at least 2")
        self.num_classes = num_classes
        self.widths = tuple(min(base_channels * 2**level, max_channels) for level in range(5))
        self.encoders = nn.ModuleList(
            _ConvBlock(1 if level == 0 else self.widths[level - 1], channels)
            for level, channels in enumerate(self.widths)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoders = nn.ModuleList(
            _UpBlock(self.widths[level], self.widths[level - 1])
            for level in range(4, 0, -1)
        )
        self.head = nn.Conv2d(self.widths[0], num_classes, kernel_size=1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, images: Tensor) -> list[Tensor]:
        """Unfused single-slice features; reusable during volume inference."""
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("UNet inputs must have shape (N, 1, H, W)")
        if min(images.shape[-2:]) < 16:
            raise ValueError("UNet inputs require H and W >= 16 for four downsampling stages")
        skips = []
        features = images
        for level, encoder in enumerate(self.encoders):
            if level:
                features = self.pool(features)
            features = encoder(features)
            skips.append(features)
        return skips

    def decode_features(self, skips: list[Tensor]) -> Tensor:
        features = skips[-1]
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            features = decoder(features, skip)
        return features

    def forward_features(self, images: Tensor) -> Tensor:
        """Last decoder features, with no additional parameters or buffers."""
        return self.decode_features(self.encode(images))

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.forward_features(images))

    def forward_with_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.forward_features(images)
        return self.head(features), features


class StudentTeacherSystem(nn.Module):
    """One or two independent students, with one frozen EMA teacher per student.

    Baselines use the same U-Net; ``dual_ordinal`` factorizes student 2 into
    tissue and relative-level heads. The caller controls their objectives.
    Only ``dual_frequency`` changes an input view: student 2
    and its teacher receive ``x + frequency_strength * high(x)``. Apply shared
    spatial augmentation before calling this module so pseudo labels remain
    aligned. Teachers never mix weights across students.
    """

    def __init__(self, cfg: Any, *, initialize_text: bool = True) -> None:
        super().__init__()
        if cfg.method not in METHODS:
            raise ValueError(f"Unknown method {cfg.method!r}; expected one of {sorted(METHODS)}")
        self.method = str(cfg.method)
        self.ema_decay = float(cfg.ema_decay)
        self.frequency_strength = float(cfg.frequency_strength)
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must be finite and between 0 and 1")
        if not math.isfinite(self.frequency_strength) or self.frequency_strength < 0:
            raise ValueError("frequency_strength must be finite and nonnegative")
        count = 2 if self.method.startswith("dual_") else 1
        text_embeddings = None
        self.text_provenance = None
        if getattr(cfg, "internal_text", "none") != "none" and initialize_text:
            from .internal_text import load_text_bank
            loader = load_text_bank
            if getattr(cfg, "text_fusion_layout", "e4") in ("hierarchical", "encoder", "encoder_multi"):
                from .hierarchical_text import load_hierarchical_bank
                loader = load_hierarchical_bank
            text_embeddings, self.text_provenance = loader(cfg.text_bank, cfg.internal_text)
        students = []
        for index in range(count):
            # Parameters are initialized on CPU, without reseeding the caller's
            # CPU state or any CUDA generator used by the data/training pipeline.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(cfg.seed) + index)
                architecture = UNet
                if self.method == "dual_ordinal" and index == 1:
                    from .ordinal_model import OrdinalUNet
                    architecture = OrdinalUNet
                extra = {}
                if getattr(cfg, "slice_context", "none") != "none":
                    from .slice_context import ContextUNet
                    architecture = ContextUNet
                    extra["mode"] = cfg.slice_context
                if getattr(cfg,'case_reference','none') != 'none' or getattr(cfg,'context_position','none') != 'none':
                    from .case_context import CaseContextUNet
                    architecture = CaseContextUNet
                    extra.update(case_reference=cfg.case_reference,context_position=cfg.context_position)
                if getattr(cfg, 'reference_condition', 'none') != 'none':
                    from .reference_conditioning import ReferenceContextUNet
                    architecture = ReferenceContextUNet
                    extra.update(reference_condition=cfg.reference_condition, reference_windows=cfg.reference_windows,
                                 reference_sources=getattr(cfg, 'reference_sources', 'case'))
                if getattr(cfg, "internal_text", "none") != "none":
                    from .internal_text import TextContextUNet
                    architecture = TextContextUNet
                    extra.update(text_embeddings=text_embeddings, text_adapter_dim=cfg.text_adapter_dim)
                    if getattr(cfg, "text_fusion_layout", "e4") == "hierarchical":
                        from .hierarchical_text import HierarchicalTextUNet
                        architecture = HierarchicalTextUNet
                        extra["text_layer_mode"] = cfg.text_layer_mode
                    elif cfg.text_fusion_layout == "encoder":
                        from .encoder_text import EncoderTextContextUNet
                        architecture = EncoderTextContextUNet
                        extra.update(text_feature_kind=cfg.text_feature_kind,
                                     encoder_text_stage=cfg.encoder_text_stage)
                    elif cfg.text_fusion_layout == "encoder_multi":
                        from .dual_text import DualTextContextUNet
                        architecture = DualTextContextUNet
                        extra.update(encoder_text_stages=cfg.encoder_text_stages,
                            text_routing_mode=cfg.text_routing_mode, text_residual_mode=cfg.text_residual_mode)
                student = architecture(
                    num_classes=int(cfg.num_classes),
                    base_channels=int(cfg.base_channels),
                    max_channels=int(cfg.max_channels),
                    **extra,
                )
            students.append(student)
        self.students = nn.ModuleList(students)
        self.teachers = nn.ModuleList(copy.deepcopy(student) for student in self.students)
        self.teachers.requires_grad_(False)
        self.teachers.eval()
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True) -> StudentTeacherSystem:
        super().train(mode)
        self.teachers.eval()
        return self

    def student_parameters(self) -> Iterator[nn.Parameter]:
        """Return only trainable student parameters for optimizer construction."""
        return self.students.parameters()

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.students):
            raise ValueError(f"Student index {index} is unavailable for {self.method}")

    def make_view(self, images: Tensor, index: int) -> Tensor:
        self._check_index(index)
        if self.method == "dual_frequency" and index == 1:
            _, high = haar_components(images)
            return images + self.frequency_strength * high
        return images

    def student_logits(self, images: Tensor, index: int, position=None) -> Tensor:
        return self.students[index](self.make_view(images, index), **({} if position is None else {'position':position}))

    @torch.no_grad()
    def teacher_logits(self, images: Tensor, index: int, position=None) -> Tensor:
        return self.teachers[index](self.make_view(images, index), **({} if position is None else {'position':position}))

    def student_outputs(self, images: Tensor, index: int, position=None) -> tuple[Tensor, Tensor]:
        return self.students[index].forward_with_features(self.make_view(images, index), **({} if position is None else {'position':position}))

    @torch.no_grad()
    def teacher_outputs(self, images: Tensor, index: int, position=None) -> tuple[Tensor, Tensor]:
        return self.teachers[index].forward_with_features(self.make_view(images, index), **({} if position is None else {'position':position}))

    @torch.no_grad()
    def predict_proba(self, images: Tensor, network: str, position=None) -> Tensor:
        """Predict with an individual model or the equal-weight probability mean."""
        if network in {"students_mean", "teachers_mean"}:
            predict = self.student_logits if network == "students_mean" else self.teacher_logits
            probabilities = [
                predict(images, index, position).float().softmax(dim=1)
                for index in range(len(self.students))
            ]
            return torch.stack(probabilities, dim=0).mean(dim=0)
        networks = {
            "student1": (self.student_logits, 0),
            "student2": (self.student_logits, 1),
            "teacher1": (self.teacher_logits, 0),
            "teacher2": (self.teacher_logits, 1),
        }
        if network not in networks:
            raise ValueError(f"Unknown prediction network {network!r}")
        predict, index = networks[network]
        self._check_index(index)
        return predict(images, index, position).float().softmax(dim=1)

    @torch.no_grad()
    def update_teachers(self) -> float:
        """Update each teacher from its paired student after an optimizer step.

        The first update copies the trained student (decay 0). Subsequent
        updates ramp to the requested EMA decay, avoiding a long contribution
        from the randomly initialized teacher. The update count is checkpointed.
        """
        decay = min(self.ema_decay, 1.0 - 1.0 / (int(self.ema_updates.item()) + 1))
        for student, teacher in zip(self.students, self.teachers):
            for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
                teacher_parameter.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
            # This U-Net has no running-statistic buffers. Copying any future
            # buffers keeps integer counters well-defined and pairing explicit.
            for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
                teacher_buffer.copy_(student_buffer)
        self.ema_updates.add_(1)
        self.teachers.eval()
        return decay
