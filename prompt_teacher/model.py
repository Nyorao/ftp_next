"""Four-way adaptation of MedCLIPSeg's PVL teacher, using native frozen CLIP blocks.

PVL equations and ScaleBlock adapted from HealthX-Lab/MedCLIPSeg (Apache-2.0;
see vendor/MedCLIPSeg-LICENSE). ScaleBlock's upstream attribution is retained:
Copyright 2025 Mobile Perception Systems Lab at TU/e, MIT License.
This implementation does not use their task-trained segmentation checkpoints.
"""
from __future__ import annotations

import math
import os
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from freqssl.prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, mc_tissue_targets, file_sha256

MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MODEL_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"


class ProbCrossAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * 2)
        self.v_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_k, self.norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(0.))
        self.beta, self.dim = 2.35, dim

    def forward(self, query, context, context_mask=None, sample=True):
        q = self.q_proj(query)
        k_mean, k_raw_var = self.k_proj(context).chunk(2, -1)
        v_mean, v_raw_var = self.v_proj(context).chunk(2, -1)
        k_mean, v_mean = self.norm_k(k_mean), self.norm_v(v_mean)
        k_var, v_var = F.softplus(k_raw_var) + 1e-6, F.softplus(v_raw_var) + 1e-6
        scores = (q @ k_mean.transpose(1, 2)) / math.sqrt(self.dim)
        variance = (q.square() @ k_var.transpose(1, 2)) / self.dim
        scores = scores - self.beta * variance.clamp_min(1e-12).sqrt()
        if context_mask is not None:
            scores = scores.masked_fill(~context_mask[:, None].bool(), torch.finfo(scores.dtype).min)
        values = v_mean + v_var.sqrt() * torch.randn_like(v_var) if sample else v_mean
        output = scores.softmax(-1) @ values
        gate = self.gate.sigmoid()
        return gate * self.out_proj(output) + (1 - gate) * query


class PVLAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_down = nn.Linear(768, 256, bias=False)
        self.text_down = nn.Linear(768, 256, bias=False)
        self.visual_up = nn.Linear(256, 768, bias=False)
        self.text_up = nn.Linear(256, 768, bias=False)
        self.image_to_text, self.text_to_image = ProbCrossAttention(), ProbCrossAttention()

    def forward(self, visual, text, text_mask, sample=True):
        visual, text = self.visual_down(visual), self.text_down(text)
        visual = self.image_to_text(visual, text, text_mask, sample)
        text = self.text_to_image(text, visual, sample=sample)
        return self.visual_up(visual), self.text_up(text)


class ScaleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        from timm.layers import LayerNorm2d
        self.layers = nn.Sequential(nn.ConvTranspose2d(512, 512, 2, 2), nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False), LayerNorm2d(512))

    def forward(self, x):
        return self.layers(x)


class TissueTeacher(nn.Module):
    def __init__(self, base, tokenizer, source, gradient_checkpointing=True):
        super().__init__()
        if source not in ("text", "neutral"):
            raise ValueError("Teacher source must be text or neutral")
        self.base = base
        self.base.requires_grad_(False)
        self.source = source
        self.prompts = TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS
        self.register_buffer("tokens", tokenizer(list(self.prompts), context_length=256), persistent=False)
        self.adapters = nn.ModuleList([PVLAdapter() for _ in range(10)])
        self.mask_head = nn.Sequential(nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))
        self.upscale = nn.Sequential(ScaleBlock(), ScaleBlock())
        self.gradient_checkpointing = gradient_checkpointing
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def adaptation_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("base.")}

    def load_adaptation(self, state):
        expected = set(self.adaptation_state())
        if set(state) != expected:
            raise ValueError("Teacher adapter checkpoint keys differ")
        self.load_state_dict(state, strict=False)

    def _stage(self, index, visual, text, mask, extended_mask, sample, use_adapters):
        if use_adapters and 1 <= index <= 10:
            dv, dt = self.adapters[index - 1](visual, text, mask, sample)
            visual, text = visual + dv, text + dt
        visual = self.base.visual.trunk.blocks[index](visual)
        text = self.base.text.transformer.encoder.layer[index](text, attention_mask=extended_mask)[0]
        return visual, text

    def encode(self, images, sample=True, use_adapters=True, prompt_tokens=None):
        batch = len(images)
        images = images[:, None].expand(-1, 4, -1, -1, -1).reshape(batch * 4, *images.shape[1:])
        tokens = (self.tokens[None].expand(batch, -1, -1) if prompt_tokens is None else prompt_tokens)
        if tokens.shape[:2] != (batch, 4):
            raise ValueError("Expected per-image four-class prompt tokens")
        tokens = tokens.reshape(batch * 4, -1)
        mask = tokens != self.base.text.config.pad_token_id
        text = self.base.text.transformer.embeddings(input_ids=tokens)
        extended_mask = (~mask[:, None, None]).to(text.dtype) * torch.finfo(text.dtype).min
        trunk = self.base.visual.trunk
        visual = trunk.norm_pre(trunk._pos_embed(trunk.patch_embed(images)))
        for index in range(12):
            def stage(v, t, index=index):
                return self._stage(index, v, t, mask, extended_mask, sample, use_adapters)
            if self.training and self.gradient_checkpointing and index >= 1 and use_adapters:
                visual, text = checkpoint(stage, visual, text, use_reentrant=False, preserve_rng_state=True)
            else:
                visual, text = stage(visual, text)
        visual = self.base.visual.head(trunk.norm(visual))
        text = self.base.text.proj(text[:, 0])
        return visual, text

    def forward(self, images, sample=True, prompt_tokens=None, return_text_features=False):
        if images.shape[1:] != (3, 224, 224):
            raise ValueError("Teacher expects normalized RGB copies with shape (B,3,224,224)")
        visual, text = self.encode(images, sample=sample, prompt_tokens=prompt_tokens)
        visual, text = F.normalize(visual, dim=-1), F.normalize(text, dim=-1)
        spatial = visual[:, 1:].transpose(1, 2).reshape(-1, 512, 14, 14)
        spatial = self.upscale(spatial)
        query = self.mask_head(text)
        logits = (spatial * query[:, :, None, None]).sum(1)
        logits = F.interpolate(logits[:, None], (224, 224), mode="bilinear", align_corners=False)
        logits = logits.reshape(len(images), 4, 224, 224)
        return (logits, text.reshape(len(images), 4, -1)) if return_text_features else logits

    @torch.inference_mode()
    def predict_mc(self, images, seed):
        device_ids = [images.device.index] if images.is_cuda else []
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(seed)
            if images.is_cuda:
                torch.cuda.manual_seed(seed)
            samples = torch.stack([self(images).float().softmax(1) for _ in range(8)])
        return mc_tissue_targets(samples)


def build_teacher(source, device, seed=42, gradient_checkpointing=True):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from open_clip import create_model_from_pretrained, get_tokenizer
    # The local hub ref is checked so an unrelated cached revision cannot silently win.
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--"))
    if (root / "refs/main").read_text().strip() != MODEL_REVISION:
        raise ValueError("BiomedCLIP cache revision differs from the frozen project revision")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base, transform = create_model_from_pretrained("hf-hub:" + MODEL_ID)
        # Reset after loading the base: both prompt arms get identical adapter initializations.
        torch.manual_seed(seed)
        model = TissueTeacher(base, get_tokenizer("hf-hub:" + MODEL_ID), source, gradient_checkpointing)
    normalize = transform.transforms[-1]
    metadata = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
        "checkpoint": str(root / "snapshots" / MODEL_REVISION / "open_clip_pytorch_model.bin"),
        "mean": list(normalize.mean), "std": list(normalize.std),
        "adapter_dim": 256, "fusion_layers_zero_based": list(range(1, 11)),
        "upscale_blocks": 2, "text_context_length": 256,
        "reference": "https://github.com/HealthX-Lab/MedCLIPSeg",
        "task_trained_checkpoint_used": False,
        "open_clip_config_sha256": file_sha256(root / "snapshots" / MODEL_REVISION / "open_clip_config.json"),
        "packages": {name: version(name) for name in ("torch", "open_clip_torch", "transformers", "timm")},
        "numeric_settings": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}
    text_root = root.parent / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"
    text_revision = (text_root / "refs/main").read_text().strip()
    metadata["tokenizer_revision"] = text_revision
    metadata["tokenizer_assets_sha256"] = {name: file_sha256(text_root / "snapshots" / text_revision / name)
        for name in ("vocab.txt", "config.json", "tokenizer_config.json")}
    return model.to(device), metadata
