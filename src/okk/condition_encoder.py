from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from okk.backbones import BackboneProjector, TimmPatchBackbone
from okk.transforms import denormalize_tensor, gaussian_blur_tensor, normalize_tensor


class LowFrequencyConditionEncoder(nn.Module):
    def __init__(self, out_dim: int = 128, grid_size: int = 16, blur_sigma: float = 2.0):
        super().__init__()
        self.grid_size = grid_size
        self.blur_sigma = blur_sigma
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, target_tokens: int | None = None) -> torch.Tensor:
        x01 = denormalize_tensor(x)
        low = gaussian_blur_tensor(x01, self.blur_sigma)
        low = F.interpolate(low, size=(self.grid_size, self.grid_size), mode="bicubic", align_corners=False)
        feat = self.net(low)
        tokens = feat.flatten(2).transpose(1, 2)
        if target_tokens is not None and tokens.shape[1] != target_tokens:
            side = int(round(target_tokens ** 0.5))
            feat = F.interpolate(feat, size=(side, side), mode="bilinear", align_corners=False)
            tokens = feat.flatten(2).transpose(1, 2)
        return tokens


class DINOConditionEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, out_dim: int = 256):
        super().__init__()
        self.backbone = TimmPatchBackbone(backbone_name, pretrained=pretrained)
        self.projector = BackboneProjector(self.backbone.feature_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, target_tokens: int | None = None) -> torch.Tensor:
        with torch.no_grad():
            patches = self.backbone(x).patches
        tokens = self.projector(patches)
        if target_tokens is not None and tokens.shape[1] != target_tokens:
            tokens = resize_tokens(tokens, target_tokens)
        return tokens


class HybridConditionEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, dino_dim: int = 256, low_dim: int = 128):
        super().__init__()
        self.dino = DINOConditionEncoder(backbone_name, pretrained=pretrained, out_dim=dino_dim)
        self.low = LowFrequencyConditionEncoder(out_dim=low_dim)
        self.out_dim = dino_dim + low_dim

    def forward(self, x: torch.Tensor, target_tokens: int | None = None) -> torch.Tensor:
        c_dino = self.dino(x, target_tokens=target_tokens)
        c_low = self.low(x, target_tokens=c_dino.shape[1])
        return torch.cat([c_dino, c_low], dim=-1)


def resize_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    b, n, c = tokens.shape
    src_side = int(round(n ** 0.5))
    dst_side = int(round(target_tokens ** 0.5))
    if src_side * src_side != n or dst_side * dst_side != target_tokens:
        return F.interpolate(tokens.transpose(1, 2), size=target_tokens, mode="linear", align_corners=False).transpose(1, 2)
    feat = tokens.transpose(1, 2).reshape(b, c, src_side, src_side)
    feat = F.interpolate(feat, size=(dst_side, dst_side), mode="bilinear", align_corners=False)
    return feat.flatten(2).transpose(1, 2)

