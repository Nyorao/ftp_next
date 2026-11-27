"""Fixed tissue-text conditioning after CRF4, before the first decoder.

Inspired by Textmatch's internal prompt decoding and SemiVL's semantic
anchors; this is a small experimental adapter, not either paper's reproduction.
No text encoder, image/text pre-alignment, or extra loss is assumed here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .prompt_supervision import TEXT_PROMPTS, NEUTRAL_PROMPTS, file_sha256
from .slice_context import ContextUNet


TEXT_DIM = 512
TISSUES = ("background", "vertebra", "canal", "disc")


def load_text_bank(path, source):
    """Load a provenance-bearing bank at training initialization only."""
    if source not in ("text", "neutral"):
        raise ValueError("Internal text source must be text or neutral")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prompts = TEXT_PROMPTS if source == "text" else NEUTRAL_PROMPTS
    if (payload.get("format_version") != 1 or payload.get("source") != source
            or payload.get("tissues") != list(TISSUES)
            or payload.get("prompts") != list(prompts)):
        raise ValueError("Text bank source, prompts or tissue ontology mismatch")
    vectors = payload.get("embeddings")
    if (not isinstance(vectors, torch.Tensor) or vectors.shape != (4, TEXT_DIM)
            or vectors.dtype != torch.float32 or not torch.isfinite(vectors).all()
            or not torch.allclose(vectors.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)):
        raise ValueError("Expected four finite unit-normalized float32 text embeddings")
    metadata = payload.get("provenance")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Text bank must record its encoder provenance")
    return vectors.detach().clone(), {k: v for k, v in payload.items() if k != "embeddings"} | {
        "file_sha256": file_sha256(path)}


class TissueTextAdapter(nn.Module):
    """Image queries attend to a frozen set of four distinct tissue vectors.

    A zero-initialized bounded gate preserves the baseline at initialization.
    Query/key/value projections learn from the existing segmentation objective.
    Token order is immaterial: permuting all keys and values together is NOT
    a valid semantic negative control for this set-attention architecture.
    """
    def __init__(self, channels, hidden_dim=64, embeddings=None):
        super().__init__()
        # None is reserved for strict checkpoint reconstruction. Forward fails
        # until valid vectors have been restored by load_state_dict.
        if embeddings is None:
            embeddings = torch.full((4, TEXT_DIM), float("nan"))
        if embeddings.shape != (4, TEXT_DIM):
            raise ValueError("Expected (4,512) embeddings")
        self.register_buffer("embeddings", embeddings.detach().float().clone())
        self.visual_norm = nn.LayerNorm(channels)
        self.text_norm = nn.LayerNorm(TEXT_DIM)
        self.query = nn.Linear(channels, hidden_dim, bias=False)
        self.key = nn.Linear(TEXT_DIM, hidden_dim, bias=False)
        self.value = nn.Linear(TEXT_DIM, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, channels, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.hidden_dim = hidden_dim

    def attention(self, features):
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("Text embeddings have not been initialized/restored")
        visual = features.flatten(2).transpose(1, 2)
        text = self.text_norm(self.embeddings)
        q = self.query(self.visual_norm(visual))
        weights = (q @ self.key(text).T / math.sqrt(self.hidden_dim)).softmax(-1)
        return weights, self.value(text)

    def forward(self, features):
        weights, values = self.attention(features)
        correction = self.output(weights @ values).transpose(1, 2).reshape_as(features)
        return features + self.gate.tanh() * correction


class TextContextUNet(ContextUNet):
    def __init__(self, num_classes, base_channels, max_channels, mode,
                 text_embeddings=None, text_adapter_dim=64):
        super().__init__(num_classes, base_channels, max_channels, mode)
        # Same visual initialization and subsequent data RNG as the control.
        with torch.random.fork_rng(devices=[]):
            self.text_adapter = TissueTextAdapter(self.widths[-1], text_adapter_dim, text_embeddings)

    def decode_features(self, skips):
        # Called by ordinary triplets AND cached volume inference after CRF.
        return super().decode_features([*skips[:-1], self.text_adapter(skips[-1])])
