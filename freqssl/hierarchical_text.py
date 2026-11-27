"""Three-stage frozen-token mixing with explicit dense tissue alignment.

The U-Net and intermediate language tokens are NOT assumed pre-aligned.
An optional global distillation head can transfer the paired foundation visual
space; global alignment alone is not evidence of dense pixel/text alignment.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .anatomy import family_labels
from .slice_context import ContextUNet
from .prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, file_sha256

TOKEN_LENGTH = 32
TEXT_WIDTH = 768
GROUPS = ((8, 9, 10, 11), (4, 5, 6, 7), (0, 1, 2, 3))


def load_hierarchical_bank(path, source):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prompts = TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS
    if (source not in ("text", "neutral") or payload.get("format_version") != 2
            or payload.get("source") != source or payload.get("prompts") != list(prompts)):
        raise ValueError("Hierarchical text bank provenance/source mismatch")
    h, mask, anchors = (payload.get(k) for k in ("hidden_tokens", "token_mask", "anchors"))
    if (not isinstance(h, torch.Tensor) or h.shape != (12, 4, TOKEN_LENGTH, TEXT_WIDTH)
            or h.dtype != torch.float32 or not torch.isfinite(h).all()
            or not isinstance(mask, torch.Tensor) or mask.shape != (4, TOKEN_LENGTH)
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or not isinstance(anchors, torch.Tensor) or anchors.shape != (4, 512)
            or not torch.isfinite(anchors).all()
            or not torch.allclose(anchors.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)):
        raise ValueError("Invalid hierarchical text bank tensors")
    if not payload.get("provenance"):
        raise ValueError("Missing encoder provenance")
    return (h, mask, anchors), {k:v for k,v in payload.items()
        if k not in ("hidden_tokens", "token_mask", "anchors")} | {"file_sha256":file_sha256(path)}


class HierarchicalTextFusion(nn.Module):
    def __init__(self, channels, dim=64, bank=None, layer_mode="grouped"):
        super().__init__()
        if layer_mode not in ("grouped", "last"):
            raise ValueError("Unknown text layer mode")
        if bank is None:
            bank = (torch.full((12,4,TOKEN_LENGTH,TEXT_WIDTH), float("nan")),
                    torch.ones(4,TOKEN_LENGTH,dtype=torch.bool), torch.full((4,512),float("nan")))
        for name, value in zip(("hidden_tokens", "token_mask", "anchors"), bank):
            self.register_buffer(name, value.detach().clone())
        self.layer_mode, self.dim = layer_mode, dim
        self.key = nn.Linear(TEXT_WIDTH, dim, bias=False)
        self.value = nn.Linear(TEXT_WIDTH, dim, bias=False)
        self.norms = nn.ModuleList(nn.LayerNorm(c) for c in channels)
        self.queries = nn.ModuleList(nn.Linear(c, dim, bias=False) for c in channels)
        self.outputs = nn.ModuleList(nn.Linear(dim, c, bias=False) for c in channels)
        self.mixers = nn.ModuleList(nn.Sequential(nn.Linear(c,16),nn.GELU(),nn.Linear(16,4)) for c in channels)
        for layer in self.mixers:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)
        self.gates = nn.Parameter(torch.zeros(3))

    def forward(self, features, stage):
        if not torch.isfinite(self.hidden_tokens).all():
            raise ValueError("Restore a valid hierarchical bank before prediction")
        indices = GROUPS[stage] if self.layer_mode == "grouped" else (11,11,11,11)
        # Padded tokens are nulled before normalization/projection, not merely
        # assigned small attention afterwards. Their values cannot leak in.
        hidden = self.hidden_tokens[list(indices)].masked_fill(~self.token_mask[None,:,:,None],0)
        hidden = F.layer_norm(hidden, (TEXT_WIDTH,))
        mix = self.mixers[stage](features.mean((-2,-1))).softmax(-1)
        text = torch.einsum("bk,kctd->bctd",mix,hidden)
        q = self.queries[stage](self.norms[stage](features.flatten(2).transpose(1,2)))
        z = torch.einsum("bnd,bctd->bnct",q,self.key(text)) / math.sqrt(self.dim)
        z = z.masked_fill(~self.token_mask[None,None],float("-inf"))
        class_logits = z.logsumexp(-1) - self.token_mask.sum(-1).float().log()
        probabilities = class_logits.softmax(-1)
        within = torch.einsum("bnct,bctd->bncd",z.softmax(-1),self.value(text))
        attended = (probabilities[...,None] * within).sum(2)
        correction = self.outputs[stage](attended).transpose(1,2).reshape_as(features)
        logits = class_logits.transpose(1,2).reshape(len(features),4,*features.shape[-2:])
        return features + self.gates[stage].tanh()*correction, logits, mix


class HierarchicalTextUNet(ContextUNet):
    def __init__(self, num_classes, base_channels, max_channels, mode,
                 text_embeddings=None, text_adapter_dim=64, text_layer_mode="grouped"):
        super().__init__(num_classes,base_channels,max_channels,mode)
        with torch.random.fork_rng(devices=[]):
            self.text_pyramid = HierarchicalTextFusion(
                [self.widths[4],self.widths[3],self.widths[2]],text_adapter_dim,text_embeddings,text_layer_mode)
            self.global_alignment = nn.Linear(self.widths[4],512,bias=False)

    def _decode(self, skips, collect=False):
        feature = skips[-1]
        global_embedding = self.global_alignment(feature.mean((-2,-1))) if collect else None
        logits, mixtures = [], []
        for index, (decoder,skip) in enumerate(zip(self.decoders,reversed(skips[:-1]))):
            if index < 3:
                feature, p, mix = self.text_pyramid(feature,index)
                if collect:
                    logits.append(p)
                    mixtures.append(mix)
            feature = decoder(feature,skip)
        return feature, dict(tissue_logits=logits, layer_mixtures=mixtures, global_embedding=global_embedding)

    def decode_features(self, skips):
        return self._decode(skips)[0]

    def forward_with_alignment(self, images):
        if not self.training:
            raise ValueError("Alignment collection uses the training path only")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected image triplets")
        encoded = self.encode(images.reshape(-1,1,*images.shape[-2:]))
        triplets = [x.reshape(len(images),3,*x.shape[1:]) for x in encoded]
        skips = [triplets[0][:,1],triplets[1][:,1]]
        skips.extend(fuse(value) for fuse,value in zip(self.context_fusion,triplets[2:]))
        feature, aux = self._decode(skips,True)
        return self.project(feature), aux


def tissue_alignment_loss(logits, labels, balance="pixel"):
    """Labeled examples only: soft area occupancy, never unlabeled GT."""
    if balance not in ("pixel", "class"):
        raise ValueError("Unknown tissue alignment balance")
    tissue = family_labels(labels)
    target = F.one_hot(tissue,4).permute(0,3,1,2).float()
    losses = []
    for p in logits:
        occupancy = F.adaptive_avg_pool2d(target,p.shape[-2:])
        weighted = -occupancy * p.float().log_softmax(1)
        if balance == "class":
            mass = occupancy.sum((0,2,3))
            # Average over tissues actually present in this labeled batch.
            # Absent tissues still compete in the softmax denominator.
            losses.append((weighted.sum((0,2,3))/mass.clamp_min(1e-6))[mass>0].mean())
        else:
            losses.append(weighted.sum(1).mean())
    return torch.stack(losses).mean()


def global_alignment_loss(embedding, target):
    if (target.shape != embedding.shape or not torch.isfinite(target).all()
            or not torch.allclose(target.norm(dim=-1),torch.ones_like(target[:,0]),atol=1e-4,rtol=1e-4)):
        raise ValueError("Expected matching unit foundation image embeddings")
    return (1-(F.normalize(embedding,dim=-1)*target.detach()).sum(-1)).mean()


def centered_global_alignment_loss(embedding, target):
    """Match image-specific variation as well as the shared biomedical mean.

    Supplementary prototype loss, not a pretrained guarantee of dense alignment.
    Requires at least two distinct image targets; the target tower is frozen.
    """
    if len(target) < 2 or embedding.shape != target.shape:
        raise ValueError("Centered alignment needs matching image pairs")
    target = target.detach()
    centered_target = target-target.mean(0,keepdim=True)
    if not torch.isfinite(target).all() or (centered_target.norm(dim=-1)<1e-6).any():
        raise ValueError("Centered alignment needs distinct finite image targets")
    centered = embedding-embedding.mean(0,keepdim=True)
    return (1-(F.normalize(centered,dim=-1)*F.normalize(centered_target,dim=-1)).sum(-1)).mean()
