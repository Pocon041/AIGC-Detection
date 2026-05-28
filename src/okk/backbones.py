from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    cls: torch.Tensor
    patches: torch.Tensor
    grid_size: tuple[int, int]


class TimmPatchBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required: pip install timm") from exc
        self.model_name = model_name
        create_kwargs = {"pretrained": pretrained, "num_classes": 0}
        try:
            self.model = timm.create_model(model_name, **create_kwargs)
        except Exception:
            if model_name.startswith("timm/"):
                self.model = timm.create_model(f"hf-hub:{model_name}", **create_kwargs)
            else:
                raise
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feature_dim = getattr(self.model, "num_features", None)
        if self.feature_dim is None:
            raise ValueError(f"unable to read num_features from model: {model_name}")

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> BackboneOutput:
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)
        else:
            feat = self.model(x)
        if isinstance(feat, dict):
            if "x_norm_patchtokens" in feat:
                patches = feat["x_norm_patchtokens"]
                cls = feat.get("x_norm_clstoken", patches.mean(dim=1))
            elif "tokens" in feat:
                tokens = feat["tokens"]
                cls = tokens[:, 0]
                patches = tokens[:, 1:]
            else:
                value = next(iter(feat.values()))
                if value.dim() == 3:
                    cls = value[:, 0]
                    patches = value[:, 1:]
                elif value.dim() == 4:
                    patches = value.flatten(2).transpose(1, 2)
                    cls = patches.mean(dim=1)
                else:
                    raise ValueError("unable to parse timm dict feature")
        elif torch.is_tensor(feat):
            if feat.dim() == 3:
                cls = feat[:, 0]
                patches = feat[:, 1:]
            elif feat.dim() == 4:
                patches = feat.flatten(2).transpose(1, 2)
                cls = patches.mean(dim=1)
            elif feat.dim() == 2:
                cls = feat
                patches = feat.unsqueeze(1)
            else:
                raise ValueError(f"unknown feature shape: {feat.shape}")
        else:
            raise ValueError(f"unknown feature type: {type(feat)}")
        grid = infer_grid_size(patches.shape[1])
        return BackboneOutput(cls=cls, patches=patches, grid_size=grid)


def infer_grid_size(num_patches: int) -> tuple[int, int]:
    side = int(round(num_patches ** 0.5))
    if side * side == num_patches:
        return side, side
    return num_patches, 1


class BackboneProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

