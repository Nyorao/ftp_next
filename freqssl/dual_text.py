"""E2 word tokens + E3 CLS with supervised routing and text-minus-zero fusion."""
import math

import torch
from torch.nn import functional as F

from .encoder_text import EncoderSpatialText, EncoderTextContextUNet, STAGES
from .hierarchical_text import TEXT_WIDTH
from .slice_context import ContextUNet

INTERVENTIONS = ("full", "zero_e2", "zero_e3", "zero_both", "class_mismatch")


class DualSpatialText(EncoderSpatialText):
    def __init__(self, channels, dim=64, bank=None, active_stages=(2, 3),
                 routing_mode="aligned", residual_mode="difference"):
        # Identical parameter order and values to the existing three-slot module.
        super().__init__(channels, dim, bank, feature_kind="none", active_stage=None)
        self.active_stages = tuple(active_stages)
        if self.active_stages not in ((), (2,), (3,), (2, 3)):
            raise ValueError("Multi-point text selects (), (2,), (3,), or (2,3)")
        if routing_mode not in ("independent", "aligned") or residual_mode not in ("plain", "difference"):
            raise ValueError("Unknown routing/residual mode")
        self.routing_mode, self.residual_mode = routing_mode, residual_mode
        self.intervention = "full"
        self.key.requires_grad_(bool(self.active_stages) and (routing_mode == "independent" or 2 in self.active_stages))
        self.value.requires_grad_(bool(self.active_stages))
        for index, stage in enumerate(STAGES):
            active = stage in self.active_stages
            for module in (self.norms[index], self.queries[index], self.fusions[index]):
                module.requires_grad_(active)
            self.gates[index].requires_grad_(active)

    def set_intervention(self, value):
        if value not in INTERVENTIONS:
            raise ValueError("Unknown text intervention")
        if value != "full" and self.training:
            raise ValueError("Text interventions are inference-only")
        if value == "class_mismatch" and self.routing_mode != "aligned":
            raise ValueError("Content mismatch requires fixed anchor-based routing")
        self.intervention = value

    def stage_text_inputs(self, stage):
        if stage == 2:
            text, mask = self.hidden_tokens[11], self.token_mask
        elif stage == 3:
            text, mask = self.hidden_tokens[11, :, :1], torch.ones_like(self.token_mask[:, :1])
        else:
            raise ValueError("No text content for this encoder stage")
        if self.intervention == "class_mismatch":
            # Fixed permutation [3,0,1,2], including masks. Anchors are untouched.
            text, mask = text.roll(1, 0), mask.roll(1, 0)
        return text, mask

    def attend(self, q, text, mask, aligned_logits):
        if not mask.any(-1).all():
            raise ValueError("Every class needs at least one valid token")
        text = F.layer_norm(text.masked_fill(~mask[..., None], 0), (TEXT_WIDTH,))
        if self.routing_mode == "aligned":
            if aligned_logits is None:
                raise ValueError("Aligned fusion must receive its supervised logits")
            probabilities = aligned_logits.softmax(1).flatten(2).transpose(1, 2)
        scores = None
        if text.shape[1] > 1 or self.routing_mode == "independent":
            scores = torch.einsum("bnd,ctd->bnct", q, self.key(text)) / math.sqrt(self.dim)
            scores = scores.masked_fill(~mask[None, None], float("-inf"))
        if self.routing_mode == "independent":
            probabilities = (scores.logsumexp(-1) - mask.sum(-1).float().log()).softmax(-1)
        token_prob = (scores.softmax(-1) if scores is not None else
                      q.new_ones((*q.shape[:2], 4, 1)))
        within = torch.einsum("bnct,ctd->bncd", token_prob, self.value(text))
        spatial = (probabilities[..., None] * within).sum(2)
        return spatial, probabilities, token_prob

    def correction(self, features, index, q, spatial, diagnostics=False):
        def pack(content):
            return torch.cat((q, content), -1).transpose(1, 2).reshape(
                len(features), 2 * self.dim, *features.shape[-2:])
        with_text = self.fusions[index](pack(spatial))
        zero_text = None
        if self.residual_mode == "difference":
            # Both calls remain on the graph. Detaching this path would break
            # cancellation of the visual-only parameter/input gradients.
            zero_text = self.fusions[index](pack(torch.zeros_like(spatial)))
            delta = with_text - zero_text
        else:
            delta = with_text
            if diagnostics:
                with torch.no_grad():
                    zero_text = self.fusions[index](pack(torch.zeros_like(spatial)))
        result = features + self.gates[index].tanh() * delta
        stats = {}
        if diagnostics:
            with torch.no_grad():
                rms = lambda x:x.detach().square().mean((1, 2, 3)).sqrt()
                denom = rms(features).clamp_min(1e-12)
                stats = dict(residual_rms_ratio=rms(result-features)/denom,
                    raw_with_text_rms_ratio=rms(with_text)/denom,
                    raw_zero_text_rms_ratio=rms(zero_text)/denom,
                    raw_difference_rms_ratio=rms(with_text-zero_text)/denom)
        return result, stats

    def forward(self, features, stage, aligned_logits=None, diagnostics=False):
        if self.training and self.intervention != "full":
            raise ValueError("Training cannot use an inference intervention")
        if stage not in self.active_stages:
            return features, {}
        index = STAGES.index(stage)
        text, mask = self.stage_text_inputs(stage)
        q = self.queries[index](self.norms[index](features.flatten(2).transpose(1, 2)))
        spatial, probabilities, token_prob = self.attend(q, text, mask, aligned_logits)
        if self.intervention in ("zero_both", f"zero_e{stage}"):
            spatial = spatial * 0
        result, stats = self.correction(features, index, q, spatial, diagnostics)
        if diagnostics:
            with torch.no_grad():
                token_entropy = -(token_prob * token_prob.clamp_min(1e-12).log()).sum(-1)
                lengths = mask.sum(-1).float().log()
                normalized = torch.where(lengths > 0, token_entropy/lengths.clamp_min(1e-12),
                                         torch.zeros_like(token_entropy))
                stats.update(class_entropy=-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean(1),
                    token_entropy=(probabilities*token_entropy).sum(-1).mean(1),
                    token_entropy_normalized=(probabilities*normalized).sum(-1).mean(1))
                for tissue in range(4):
                    stats[f"route_probability_{tissue}"] = probabilities[..., tissue].mean(1)
                    stats[f"route_selected_fraction_{tissue}"] = (probabilities.argmax(-1)==tissue).float().mean(1)
                if aligned_logits is not None:
                    aligned = aligned_logits.softmax(1).flatten(2).transpose(1, 2)
                    stats["route_aux_argmax_agreement"] = (aligned.argmax(-1)==probabilities.argmax(-1)).float().mean(1)
        return result, stats


class DualTextContextUNet(EncoderTextContextUNet):
    def __init__(self, num_classes, base_channels, max_channels, mode, text_embeddings=None,
                 text_adapter_dim=64, encoder_text_stages=(2, 3),
                 text_routing_mode="aligned", text_residual_mode="difference"):
        # Reuse the fixed-microbatch encoder wrapper and context decoder, without
        # creating/reinitializing a legacy text module first.
        ContextUNet.__init__(self, num_classes, base_channels, max_channels, mode)
        with torch.random.fork_rng(devices=[]):
            self.encoder_text = DualSpatialText([self.widths[s] for s in STAGES], text_adapter_dim,
                text_embeddings, encoder_text_stages, text_routing_mode, text_residual_mode)

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
                aligned = None
                required = level in self.encoder_text.active_stages and self.encoder_text.routing_mode == "aligned"
                if collect or required:
                    # The same pre-injection logits feed fusion and supervision.
                    # All slices are processed identically; only labeled centers
                    # enter the loss in the engine, never neighbor/unlabeled GT.
                    aligned = self.encoder_text.alignment_logits(features, STAGES.index(level))
                if collect:
                    logits.append(aligned[1::3])
                features, current = self.encoder_text(features, level, aligned, diagnostics)
                stats.update({f"E{level}_{name}": value[1::3] for name,value in current.items()})
            skips.append(features)
        return skips, dict(tissue_logits=logits, alignment_stages=["E2", "E3", "E4"],
                           encoder_text_diagnostics=stats)
