from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_ranking_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    real_image = real_scores.mean(dim=1)
    fake_image = fake_scores.mean(dim=1)
    return F.relu(margin - real_image + fake_image).mean()


def patch_bce_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    real_targets = torch.ones_like(real_scores)
    fake_targets = torch.zeros_like(fake_scores)
    logits = torch.cat([real_scores, fake_scores], dim=0)
    targets = torch.cat([real_targets, fake_targets], dim=0)
    return F.binary_cross_entropy_with_logits(logits, targets)


def image_bce_loss(real_image_scores: torch.Tensor, fake_image_scores: torch.Tensor) -> torch.Tensor:
    logits = torch.cat([real_image_scores, fake_image_scores], dim=0)
    targets = torch.cat([torch.ones_like(real_image_scores), torch.zeros_like(fake_image_scores)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, targets)


def cross_covariance_penalty(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if a.shape[:-1] != b.shape[:-1]:
        raise ValueError(f"cross-cov inputs need matching observation axes: {a.shape}, {b.shape}")
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    if a_flat.shape[0] < 2:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    a_flat = a_flat - a_flat.mean(dim=0, keepdim=True)
    b_flat = b_flat - b_flat.mean(dim=0, keepdim=True)
    a_flat = a_flat / a_flat.std(dim=0, keepdim=True).clamp_min(eps)
    b_flat = b_flat / b_flat.std(dim=0, keepdim=True).clamp_min(eps)
    cov = a_flat.transpose(0, 1).matmul(b_flat) / float(a_flat.shape[0] - 1)
    return cov.pow(2).mean()

