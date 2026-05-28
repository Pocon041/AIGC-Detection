from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class SRMHighPass(nn.Module):
    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
        k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
        weight = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)
        weight = weight.repeat(3, 1, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=2, groups=3)


class LNPResNetExtractor(nn.Module):
    def __init__(self, arch: str = "resnet18", feature_dim: int = 512, num_classes: int = 2, use_srm: bool = True):
        super().__init__()
        self.use_srm = use_srm
        self.srm = SRMHighPass() if use_srm else nn.Identity()
        in_channels = 9 if use_srm else 3
        if arch == "resnet18":
            base = resnet18(weights=None)
            backbone_dim = 512
        elif arch == "resnet34":
            base = resnet34(weights=None)
            backbone_dim = 512
        else:
            raise ValueError(f"涓嶆敮鎸佺殑 LNP ResNet: {arch}")
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.proj = nn.Conv2d(backbone_dim, feature_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_srm:
            x = self.srm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fmap = self.proj(x)
        pooled = self.pool(fmap).flatten(1)
        return {"feature_map": fmap, "pooled": pooled}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(x)
        logits = self.classifier(feat["pooled"])
        feat["logits"] = logits
        return feat


class LNPFeatureAdapter(nn.Module):
    def __init__(self, checkpoint: str, arch: str = "resnet18", feature_dim: int = 256, out_dim: int = 256, use_srm: bool = True):
        super().__init__()
        self.encoder = LNPResNetExtractor(arch=arch, feature_dim=feature_dim, num_classes=2, use_srm=use_srm)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        self.encoder.load_state_dict(state, strict=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.forward_features(x)["feature_map"]
        tokens = feat.flatten(2).transpose(1, 2)
        return self.proj(tokens)

