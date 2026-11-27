"""Case-isolated, SI-windowed image/mask conditioning after the original CRF.

Spatial axes are (AP, SI). Coordinates are normalized *unaugmented* SI pixel
centres, transported with the image, not row numbers in the augmented tensor.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import _group_count
from .slice_context import ContextUNet


def si_coordinates(batch, shape, device=None):
    h, w = shape
    y = (torch.arange(w, device=device, dtype=torch.float32) + .5) / w
    return y[None, None, None].expand(batch, 1, h, w).clone()


def row_prior(probabilities):
    return probabilities.mean(-2, keepdim=True).expand_as(probabilities)


def spatial_resize(value, shape):
    return F.interpolate(value, size=shape, mode="area")


def position_encoding(y, width=128):
    frequency = torch.exp(torch.arange(0, width, 2, device=y.device) * (-math.log(10000.) / width))
    phase = y[..., None] * (2 * math.pi) * frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


class MaskEncoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        layers = []
        previous = 21
        for channels in (16, 32, 64, 128):
            layers.append(nn.Sequential(
                nn.Conv2d(previous, channels, 3, 2, 1, bias=False),
                nn.GroupNorm(_group_count(channels), channels), nn.LeakyReLU(.01),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_group_count(channels), channels), nn.LeakyReLU(.01)))
            previous = channels
        self.layers = nn.ModuleList(layers)
        self.outputs = nn.ModuleList(nn.Conv2d(c, out, 1) for c, out in zip((64, 128), output_channels))

    def forward(self, image, probabilities):
        x, values = torch.cat((image, probabilities), 1), []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i >= 2:
                values.append(self.outputs[i - 2](x))
        return values


class RowFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.adapter = nn.Sequential(nn.Conv2d(20, 32, 1), nn.ReLU(), nn.Conv2d(32, 2 * channels, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, query, prior, valid):
        alpha, beta = self.adapter(prior).chunk(2, 1)
        return (query * alpha + beta) * valid


class RowWindowAttention(nn.Module):
    """Gather only eligible keys; the dense path exists solely as a test oracle."""
    def __init__(self, channels, window, position_only=False, dimension=128, heads=4):
        super().__init__()
        self.window, self.position_only = window, position_only
        self.dimension, self.heads, self.head_dim = dimension, heads, dimension // heads
        self.query_norm, self.key_norm = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.query = nn.Linear(channels, dimension, bias=False)
        self.key = nn.Linear(channels, dimension, bias=False)
        self.value = nn.Linear(channels, dimension, bias=False)
        self.output = nn.Conv2d(dimension, channels, 1)
        self.gate = nn.Parameter(torch.zeros(()))
        bias = -.5 * (torch.arange(-window, window + 1).float() / window).square()
        self.row_bias = nn.Parameter(bias[None].repeat(heads, 1))
        self.observe = False
        self.observations = []
        self.attention_macs = 0

    def _bias(self, delta):
        index = (delta + self.window).clamp(0, 2 * self.window)
        lower = index.floor().long()
        upper = (lower + 1).clamp_max(2 * self.window)
        fraction = index - lower
        return self.row_bias[:, lower] * (1 - fraction) + self.row_bias[:, upper] * fraction

    def attend(self, q, k, v, qy, ky, qvalid, kvalid, rows, dense=False):
        # Each batch element owns a reference bank. No cross-patient softmax.
        outputs, observations = [], []
        for batch in range(len(q)):
            order = torch.argsort(ky[batch], stable=True)
            sy = ky[batch, order]
            query_y = qy[batch]
            if dense:
                indices = order[None].expand(len(query_y), -1)
                slots = torch.ones_like(indices, dtype=torch.bool)
            else:
                radius = (self.window + 1e-5) / rows
                left = torch.searchsorted(sy.contiguous(), (query_y - radius).contiguous())
                right = torch.searchsorted(sy.contiguous(), (query_y + radius).contiguous(), right=True)
                width = max(1, int((right - left).max()))
                offset = torch.arange(width, device=q.device)[None]
                slots = offset < (right - left)[:, None]
                indices = order[(left[:, None] + offset).clamp_max(len(sy) - 1)]
            delta = (ky[batch, indices] - query_y[:, None]) * rows
            legal = slots & (delta.abs() <= self.window + 1e-5)
            legal &= (kvalid[batch, indices] > .999) & (qvalid[batch, :, None] > .999)
            key = k[batch, indices].reshape(len(query_y), -1, self.heads, self.head_dim)
            value = v[batch, indices].reshape_as(key)
            query = q[batch].reshape(-1, self.heads, self.head_dim)
            logits = torch.einsum("qhd,qkhd->hqk", query, key) / math.sqrt(self.head_dim)
            logits = logits + self._bias(delta)
            # Avoid NaNs when an entire query is padding or has no legal key.
            logits = logits.masked_fill(~legal[None], -1e9)
            weights = logits.softmax(-1) * legal[None]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            output = torch.einsum("hqk,qkhd->qhd", weights, value).flatten(1)
            outputs.append(output)
            self.attention_macs += 2 * len(query_y) * indices.shape[1] * self.dimension
            if self.observe:
                with torch.no_grad():
                    bins = delta.round().long().clamp(-self.window, self.window) + self.window
                    mass = weights.new_zeros(self.heads, 2 * self.window + 1)
                    uniform = legal.to(weights.dtype) / legal.sum(-1, keepdim=True).clamp_min(1)
                    null_mass = weights.new_zeros(2 * self.window + 1)
                    null_mass.scatter_add_(0, bins.flatten(), uniform.flatten())
                    for head in range(self.heads):
                        mass[head].scatter_add_(0, bins.flatten(), weights[head].flatten())
                    n = int(legal.any(-1).sum())
                    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
                    observations.append(dict(mass=mass.cpu().tolist(), uniform_mass=null_mass.cpu().tolist(),
                        queries=n, mean_abs_delta=(weights * delta.abs()[None]).sum((1, 2)).cpu().tolist(),
                        entropy_sum=entropy.sum(-1).cpu().tolist()))
        if self.observe:
            self.observations.extend(observations)
        return torch.stack(outputs)

    def forward(self, feature, key_image, value_mask, qsi, ksi, qvalid, kvalid, dense=False):
        b, c, h, w = feature.shape
        flatten = lambda x: x.flatten(2).transpose(1, 2)
        qy, ky = qsi.flatten(1), ksi.flatten(1)
        q = self.query(self.query_norm(flatten(feature))) + position_encoding(qy * w, self.dimension)
        content = self.key(self.key_norm(flatten(key_image)))
        k = content * (0. if self.position_only else 1.) + position_encoding(ky * w, self.dimension)
        v = self.value(flatten(value_mask))
        result = self.attend(q, k, v, qy, ky, qvalid.flatten(1), kvalid.flatten(1), w, dense)
        residual = self.gate.tanh() * self.output(result.transpose(1, 2).reshape(b, self.dimension, h, w))
        # output has a bias; no eligible bank must still produce exactly zero.
        with torch.no_grad():
            key_y = ky.masked_fill(kvalid.flatten(1) <= .999, float('inf'))
            distance = (qy[:, :, None] - key_y[:, None]).abs().amin(-1)
            available = ((distance * w <= self.window + 1e-5) & (qvalid.flatten(1) > .999)).reshape(b, 1, h, w)
        residual = residual * available
        if self.observe:
            self.observations[-b:] = [dict(obs, gate=float(self.gate.detach().tanh()),
                residual_rms_ratio=float(residual[i].detach().square().mean().sqrt() /
                    feature[i].detach().square().mean().sqrt().clamp_min(1e-8)))
                for i, obs in enumerate(self.observations[-b:])]
        return residual


class ReferenceContextUNet(ContextUNet):
    def __init__(self, num_classes, base_channels, max_channels, mode="adaptive",
                 reference_condition="kv", reference_windows=(2, 1)):
        super().__init__(num_classes, base_channels, max_channels, mode)
        self.reference_condition = reference_condition
        self.reference_teacher_path = ""
        self.reference_teacher_sha256 = ""
        channels = [self.widths[k] for k in (3, 4)]
        with torch.random.fork_rng(devices=[]):
            self.mask_encoder = None if reference_condition == "row" else MaskEncoder(channels)
            self.reference_adapters = nn.ModuleList(
                RowFiLM(c) if reference_condition == "row" else
                RowWindowAttention(c, w, reference_condition == "position")
                for c, w in zip(channels, reference_windows))

    def prepare_reference(self, image, probabilities, si, valid, prior=None, encoded=None):
        if image.ndim != 4 or image.shape[1] != 1 or probabilities.shape != (len(image), 20, *image.shape[-2:]):
            raise ValueError("Reference image/probability shape mismatch")
        if si.shape != image.shape or valid.shape != image.shape:
            raise ValueError("Reference coordinates/validity must align with its image")
        if not all(torch.isfinite(v).all() for v in (image, probabilities, si, valid)):
            raise ValueError("Nonfinite reference input")
        if self.mask_encoder is None:
            if prior is None:
                prior = row_prior(probabilities)
            shapes = [(image.shape[-2] // s, image.shape[-1] // s) for s in (8, 16)]
            keys = values = [None, None]
        else:
            keys = (self.encode(image) if encoded is None else encoded)[3:]
            if self.training:
                values = self.mask_encoder(image, probabilities)
            else:
                chunks = []
                for start in range(0, len(image), self.inference_microbatch):
                    im, pr = image[start:start+self.inference_microbatch], probabilities[start:start+self.inference_microbatch]
                    n = len(im)
                    im = self._pad(im); pr = self._pad(pr)
                    chunks.append([v[:n] for v in self.mask_encoder(im, pr)])
                values = [torch.cat([chunk[k] for chunk in chunks]) for k in range(2)]
            shapes = [v.shape[-2:] for v in keys]
        return [dict(key=key, value=value, si=spatial_resize(si, shape),
                     valid=spatial_resize(valid, shape),
                     prior=spatial_resize(prior, shape) if prior is not None else None)
                for key, value, shape in zip(keys, values, shapes)]

    @staticmethod
    def select_bank(bank, indices):
        return [{key: None if value is None else value[indices] for key, value in level.items()} for level in bank]

    def _pad(self, value):
        if len(value) < self.inference_microbatch:
            value = torch.cat((value, value.new_zeros((self.inference_microbatch-len(value), *value.shape[1:]))))
        return value

    def decode_reference(self, triplets, bank, query_si, query_valid, donor=None, mix=None):
        skips = [triplets[0][:, 1], triplets[1][:, 1]]
        skips.extend(fuse(value) for fuse, value in zip(self.context_fusion, triplets[2:]))
        if bank is None:
            return self.decode_features(skips)
        for offset, level in enumerate((3, 4)):
            original = skips[level]
            shape = original.shape[-2:]
            qsi, qvalid = spatial_resize(query_si, shape), spatial_resize(query_valid, shape)
            def residual(reference, si, valid):
                if self.reference_condition == "row":
                    support = ((reference["valid"] > .999) & (valid > .999)).to(original.dtype)
                    return self.reference_adapters[offset](original, reference["prior"], support)
                return self.reference_adapters[offset](original, reference["key"], reference["value"],
                    si, reference["si"], valid, reference["valid"])
            own = residual(bank[offset], qsi, qvalid)
            if donor is not None and mix is not None and bool(mix.any()):
                other = {key: None if value is None else value[donor] for key, value in bank[offset].items()}
                borrowed = residual(other, qsi[donor], qvalid[donor])
                mask = spatial_resize(mix, shape)
                own = own * (1 - mask) + borrowed * mask
            skips[level] = original + own
        return self.decode_features(skips)

    def forward_features(self, images, reference=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Reference model requires an unchanged local triplet")
        encoded = self.encode(images.reshape(-1, 1, *images.shape[-2:]))
        triplets = [v.reshape(len(images), 3, *v.shape[1:]) for v in encoded]
        if reference is None:
            return self.context_features(triplets)
        bank = self.prepare_reference(reference["image"], reference["probabilities"],
            reference["si"], reference["valid"], reference.get("prior"))
        if not self.training:
            if reference.get('donor') is not None:
                raise ValueError('CutMix routing is training-only')
            results = []
            for start in range(0, len(images), self.inference_microbatch):
                end = min(len(images), start+self.inference_microbatch)
                local = [{k: None if v is None else self._pad(v[start:end]) for k, v in level.items()} for level in bank]
                results.append(self.decode_reference([self._pad(v[start:end]) for v in triplets], local,
                    self._pad(reference['si'][start:end]), self._pad(reference['valid'][start:end]))[:end-start])
            return torch.cat(results)
        return self.decode_reference(triplets, bank, reference["si"], reference["valid"],
                                     reference.get("donor"), reference.get("mix"))

    def forward(self, images, reference=None):
        return self.project(self.forward_features(images, reference))

    def forward_with_features(self, images, reference=None):
        features = self.forward_features(images, reference)
        return self.project(features), features
