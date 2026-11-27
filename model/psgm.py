"""PSGM extracted from the original model/convnext_pixel.py.

The default 768 -> 192 -> 48 -> 20 dimensions and fusion operations are
unchanged. Class embeddings are buffers so device moves and checkpoints work.
"""

import torch
from torch import nn


class Conv_BN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class MLP_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, hidden_channels)
        self.l2 = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.l2(self.activation(self.l1(x)))


class PSGM(nn.Module):
    def __init__(self, class_embeddings):
        super().__init__()
        num_classes, channels = class_embeddings.shape
        self.hidden_channels = channels // 4
        self.en_fea_conv = Conv_BN_Block(channels, self.hidden_channels)
        self.fusion_mlp = MLP_Block(self.hidden_channels, channels // 16, num_classes)
        self.fusion_conv = Conv_BN_Block(2 * channels, channels)
        self.register_buffer("txt_emb", class_embeddings.detach().clone())

    def forward(self, x):
        bs, _, h, w = x.shape
        img_fea = self.en_fea_conv(x)
        txt_emb = self.txt_emb.unsqueeze(2).unsqueeze(2).unsqueeze(0)
        txt_emb = txt_emb.repeat(bs, 1, 1, h, w).permute(0, 2, 3, 4, 1)
        fusion_weight = img_fea.view(bs, self.hidden_channels, h * w).permute(0, 2, 1)
        fusion_weight = self.fusion_mlp(fusion_weight).view(bs, h, w, -1)
        fusion_weight = fusion_weight.unsqueeze(1).softmax(-1)
        txt_emb = (txt_emb * fusion_weight).sum(-1)
        return self.fusion_conv(torch.cat((x, txt_emb), dim=1))
