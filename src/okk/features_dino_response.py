from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from okk.backbones import TimmPatchBackbone
from okk.transforms import apply_tensor_perturbation, default_perturbation_modes


class DINOResponseExtractor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, modes: Dict[str, str] | None = None):
        super().__init__()
        self.backbone = TimmPatchBackbone(backbone_name, pretrained=pretrained)
        self.modes = modes or default_perturbation_modes()
        self.feature_dim = self.backbone.feature_dim
        self.response_dim = len(self.modes) * 4

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        base = self.backbone(x)
        base_patches = F.normalize(base.patches, dim=-1)
        stats: List[torch.Tensor] = []
        patch_maps: List[torch.Tensor] = []
        for _, mode in self.modes.items():
            x_t = apply_tensor_perturbation(x, mode)
            pert = self.backbone(x_t)
            pert_patches = F.normalize(pert.patches, dim=-1)
            cos = (base_patches * pert_patches).sum(dim=-1)
            delta = (base.patches - pert.patches).norm(dim=-1)
            one_minus_cos = 1.0 - cos
            stat = torch.stack([
                one_minus_cos.mean(dim=1),
                one_minus_cos.std(dim=1),
                torch.quantile(one_minus_cos, q=0.90, dim=1),
                delta.mean(dim=1),
            ], dim=1)
            stats.append(stat)
            patch_maps.append(one_minus_cos.unsqueeze(-1))
        image_features = torch.cat(stats, dim=1)
        patch_response = torch.cat(patch_maps, dim=-1)
        return {
            "image_features": image_features,
            "patch_response": patch_response,
            "base_patches": base.patches,
            "base_cls": base.cls,
            "grid_size": torch.tensor(base.grid_size, device=x.device),
        }


class ResponseProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

