from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from okk.beyond_lowlevel import BeyondLowLevelAdapter
from okk.lnp_model import LNPResNetExtractor
from okk.transforms import denormalize_tensor, gaussian_blur_tensor


class OnlineLNPLowLevelEncoder(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        arch: str = "resnet18",
        feature_dim: int = 256,
        out_dim: int = 256,
        denoise_mode: str = "gaussian",
        gain: float = 8.0,
    ):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        feature_dim = int(ckpt.get("feature_dim", feature_dim))
        self.encoder = LNPResNetExtractor(arch=arch, feature_dim=feature_dim, num_classes=2, use_srm=True)
        self.encoder.load_state_dict(ckpt["model"], strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim
        self.denoise_mode = denoise_mode
        self.gain = gain

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def make_lnp(self, x: torch.Tensor) -> torch.Tensor:
        x01 = denormalize_tensor(x)
        if self.denoise_mode == "gaussian":
            den = gaussian_blur_tensor(x01, sigma=1.0)
        elif self.denoise_mode == "avg":
            den = F.avg_pool2d(x01, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"unknown online LNP denoise mode: {self.denoise_mode}")
        lnp = (x01 - den) * self.gain + 0.5
        lnp = lnp.clamp(0.0, 1.0)
        return lnp * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lnp = self.make_lnp(x)
        with torch.no_grad():
            fmap = self.encoder.forward_features(lnp)["feature_map"]
        tokens = fmap.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class BeyondPretextLowLevelEncoder(BeyondLowLevelAdapter):
    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

