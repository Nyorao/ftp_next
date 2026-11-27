"""Randomly initialized U-Net student and its structurally identical EMA teacher."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import APGM
from .psgm import PSGM


def bilinear_resize(x, size):
    """Keep bilinear interpolation when CUDA backward is not deterministic.

    PyTorch 2.0's CUDA bilinear backward uses nondeterministic accumulation.
    For reproducible comparisons, the same operation runs on CPU with autograd
    through the device copies. This costs transfers but changes no parameters
    or interpolation mode. Inference and non-strict runs stay on the device.
    """
    if (x.is_cuda and x.requires_grad and torch.is_grad_enabled()
            and torch.are_deterministic_algorithms_enabled()):
        return F.interpolate(x.cpu(), size=size, mode="bilinear", align_corners=False).to(x.device)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Five downsampling stages retain the original 7x7 prompt grid at 224px."""

    def __init__(self, num_classes, base_channels=32, bottleneck_channels=768):
        super().__init__()
        widths = [base_channels * m for m in (1, 2, 4, 8)]
        widths += [bottleneck_channels // 2, bottleneck_channels]
        self.stem = DoubleConv(2, widths[0])
        self.down = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(left, right))
            for left, right in zip(widths, widths[1:])
        ])
        self.up = nn.ModuleList([
            DoubleConv(widths[i] + widths[i - 1], widths[i - 1])
            for i in range(len(widths) - 1, 0, -1)
        ])
        self.out = nn.Conv2d(widths[0], num_classes, 1)

    def encode(self, image):
        skips = [self.stem(image)]
        for down in self.down:
            skips.append(down(skips[-1]))
        return skips

    def decode(self, skips, bottleneck=None):
        x = skips[-1] if bottleneck is None else bottleneck
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = bilinear_resize(x, skip.shape[-2:])
            x = block(torch.cat((x, skip), dim=1))
        return self.out(x)


class FTPStudent(nn.Module):
    """One shared U-Net processes LF/HF views; each keeps its original prompt route.

    APGM and PSGM have separate parameters. The teacher is a copy of this entire
    student, not one of the two different prompt routes.
    """

    def __init__(self, text_features, class_embeddings, base_channels=32,
                 use_apgm=True, use_psgm=True):
        super().__init__()
        if text_features.ndim != 3 or text_features.shape[0] != 1:
            raise ValueError("text_features must have shape [1, text_length, text_channels]")
        if class_embeddings.ndim != 2:
            raise ValueError("class_embeddings must have shape [num_classes, channels]")
        num_classes, channels = class_embeddings.shape
        if channels < 16 or channels % 4:
            raise ValueError("Bottleneck channels must be >=16 and divisible by four")
        self.num_classes = num_classes
        self.use_apgm = use_apgm
        self.use_psgm = use_psgm
        self.unet = UNet(num_classes, base_channels, channels)
        self.register_buffer("text_features", text_features.detach().clone())
        self.register_buffer("class_embeddings", class_embeddings.detach().clone())
        text_length = text_features.shape[1]
        self.apgm = APGM(channels, text_length, text_length, text_features.shape[2]) if use_apgm else None
        self.psgm = PSGM(class_embeddings) if use_psgm else None

    def forward(self, low_view, high_view, return_plain=True):
        if low_view.shape != high_view.shape or low_view.ndim != 4 or low_view.shape[1] != 2:
            raise ValueError("LF/HF views must have the same [batch, 2, height, width] shape")
        if min(low_view.shape[-2:]) < 64:
            raise ValueError("Input height and width must be at least 64")
        batch_size = low_view.shape[0]
        # Batch the two frequency views so encoder/decoder normalization sees both.
        skips = self.unet.encode(torch.cat((low_view, high_view), dim=0))
        low, high = skips[-1].split(batch_size)
        if self.apgm is not None:
            height, width = low.shape[-2:]
            tokens = low.flatten(2).transpose(1, 2)
            low = self.apgm(tokens, self.text_features.expand(batch_size, -1, -1))
            low = low.transpose(1, 2).reshape(batch_size, -1, height, width)
        if self.psgm is not None:
            high = self.psgm(high)
        prompted = self.unet.decode(skips, torch.cat((low, high), dim=0))
        low_prompt, high_prompt = prompted.split(batch_size)
        outputs = {"lf_prompt": low_prompt, "hf_prompt": high_prompt}
        if return_plain:
            plain = self.unet.decode(skips) if self.use_apgm or self.use_psgm else prompted
            outputs["lf_plain"], outputs["hf_plain"] = plain.split(batch_size)
        return outputs


class MeanTeacher(nn.Module):
    def __init__(self, student, ema_decay=0.99):
        super().__init__()
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.ema_decay = ema_decay
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        # No teacher dropout, batch-stat updates or backpropagation.
        self.teacher.eval()
        return self

    def forward(self, low_view, high_view, return_plain=True):
        return self.student(low_view, high_view, return_plain=return_plain)

    @torch.no_grad()
    def teacher_predictions(self, low_view, high_view):
        return self.teacher(low_view, high_view, return_plain=False)

    @torch.no_grad()
    def update_teacher(self):
        # Warm-start averaging avoids excessive lag behind random initialization.
        decay = min(self.ema_decay, 1.0 - 1.0 / (int(self.ema_updates) + 1))
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.mul_(decay).add_(student, alpha=1.0 - decay)
        # Use the student's BN running statistics; average trainable weights only.
        # Fixed text buffers and integer counters are copied exactly as well.
        for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
            teacher.copy_(student)
        self.ema_updates.add_(1)
        return decay
