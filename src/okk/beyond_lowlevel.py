from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from okk.transforms import denormalize_tensor


class SRMHighPassBank(nn.Module):
    def __init__(self):
        super().__init__()
        k1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0
        k2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ], dtype=torch.float32) / 12.0
        k3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 2.0
        kernels = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        weight = self.kernels.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=2, groups=c)


class ConvGroup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeyondLowLevelExtractor(nn.Module):
    def __init__(
        self,
        arch: str = "vit_srm",
        image_size: int = 224,
        patch_size: int = 16,
        feature_dim: int = 256,
        num_pretext_classes: int = 11,
        conv_channels: int = 64,
        conv_depth: int = 3,
        transformer_depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_highpass: bool = True,
    ):
        super().__init__()
        if arch != "vit_srm":
            raise ValueError(f"unsupported low-level arch: {arch}")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.arch = arch
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_pretext_classes = num_pretext_classes
        self.use_highpass = use_highpass
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.srm = SRMHighPassBank()
        in_channels = 9 if use_highpass else 3
        groups = []
        for i in range(conv_depth):
            groups.append(ConvGroup(in_channels if i == 0 else conv_channels, conv_channels))
        self.conv_groups = nn.Sequential(*groups)
        self.patch_pool = nn.AdaptiveAvgPool2d(1)
        self.patch_proj = nn.Linear(conv_channels, feature_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.norm = nn.LayerNorm(feature_dim)
        self.image_head = nn.Linear(feature_dim, num_pretext_classes)
        self.regression_head = nn.Linear(feature_dim, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def model_config(self) -> Dict:
        return {
            "arch": self.arch,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "feature_dim": self.feature_dim,
            "num_pretext_classes": self.num_pretext_classes,
            "conv_channels": self.conv_groups[-1].net[0].out_channels,
            "conv_depth": len(self.conv_groups),
            "transformer_depth": len(self.self_attention.layers),
            "num_heads": self.self_attention.layers[0].self_attn.num_heads,
            "dropout": self.self_attention.layers[0].dropout.p,
            "use_highpass": self.use_highpass,
        }

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"input size must be divisible by patch_size: {(h, w)}, patch_size={p}")
        gh, gw = h // p, w // p
        patches = F.unfold(x, kernel_size=p, stride=p)
        patches = patches.transpose(1, 2).reshape(b * gh * gw, c, p, p)
        return patches, (gh, gw)

    def get_pos_embed(self, grid_size: tuple[int, int]) -> torch.Tensor:
        if grid_size == self.grid_size:
            return self.pos_embed
        gh0, gw0 = self.grid_size
        pos = self.pos_embed.reshape(1, gh0, gw0, self.feature_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=grid_size, mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], self.feature_dim)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x01 = denormalize_tensor(x).clamp(0.0, 1.0)
        b = x01.shape[0]
        patches, grid_size = self.patchify(x01)
        if self.use_highpass:
            patches = self.srm(patches)
        h = self.conv_groups(patches)
        tokens = self.patch_pool(h).flatten(1)
        tokens = self.patch_proj(tokens)
        tokens = tokens.reshape(b, grid_size[0] * grid_size[1], self.feature_dim)
        tokens = tokens + self.get_pos_embed(grid_size).to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.self_attention(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        feature_map = tokens.reshape(b, grid_size[0], grid_size[1], self.feature_dim).permute(0, 3, 1, 2).contiguous()
        return {"tokens": tokens, "feature_map": feature_map, "pooled": pooled, "grid_size": grid_size}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(x)
        feat["image_logits"] = self.image_head(feat["pooled"])
        feat["regression"] = self.regression_head(feat["pooled"]).squeeze(-1)
        return feat


def make_pretext_variants(x: torch.Tensor, modes: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
    raise ValueError("paper-version Beyond low-level pretext requires precomputed diffusion-denoised variants; run train_lowlevel_precomputed.py")


class BeyondLowLevelAdapter(nn.Module):
    def __init__(self, checkpoint: str, arch: str = "vit_srm", feature_dim: int = 256, out_dim: int = 256):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        model_config = ckpt.get("model_config", {})
        if not model_config:
            modes = ckpt.get("pretext_modes", ["original"])
            model_config = {
                "arch": ckpt.get("arch", arch),
                "feature_dim": int(ckpt.get("feature_dim", feature_dim)),
                "num_pretext_classes": len(modes),
            }
        self.encoder = BeyondLowLevelExtractor(**model_config)
        self.encoder.load_state_dict(ckpt["model"], strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        feature_dim = int(model_config.get("feature_dim", feature_dim))
        if int(out_dim) != feature_dim:
            raise ValueError(
                "BeyondLowLevelAdapter no longer creates an untrained projection head. "
                f"Requested out_dim={out_dim}, but checkpoint feature_dim={feature_dim}. "
                "Use the checkpoint feature_dim or train and load an explicit projection module."
            )
        self.proj = nn.Identity()
        self.out_dim = feature_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.encoder.forward_features(x)["tokens"]
        return self.proj(tokens)

