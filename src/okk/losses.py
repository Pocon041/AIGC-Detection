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

