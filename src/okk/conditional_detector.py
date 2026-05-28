from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalResidualDetector(nn.Module):
    def __init__(self, condition_dim: int, residual_dim: int, hidden_dim: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = condition_dim + residual_dim
        for i in range(depth):
            layers.append(nn.LayerNorm(in_dim if i == 0 else hidden_dim))
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, condition: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if condition.shape[1] != residual.shape[1]:
            raise ValueError(f"condition token count != residual token count: {condition.shape}, {residual.shape}")
        x = torch.cat([condition, residual], dim=-1)
        h = self.trunk(x)
        return self.head(h).squeeze(-1)


class ConditionalGaussianDetector(nn.Module):
    """Score residual tokens with a diagonal Gaussian p(R | C)."""

    def __init__(
        self,
        condition_dim: int,
        residual_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.1,
        min_logvar: float = -4.0,
        max_logvar: float = 2.0,
        mean_scale: float = 3.0,
        normalize_residual: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = condition_dim
        for i in range(depth):
            layers.append(nn.LayerNorm(in_dim if i == 0 else hidden_dim))
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, residual_dim * 2))
        self.conditioner = nn.Sequential(*layers)
        self.residual_norm = nn.LayerNorm(residual_dim, elementwise_affine=False) if normalize_residual else nn.Identity()
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.mean_scale = float(mean_scale)

    def distribution(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.conditioner(condition)
        mean_raw, logvar_raw = params.chunk(2, dim=-1)
        mean = torch.tanh(mean_raw) * self.mean_scale
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * torch.sigmoid(logvar_raw)
        return mean, logvar

    def forward(self, condition: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if condition.shape[1] != residual.shape[1]:
            raise ValueError(f"condition token count != residual token count: {condition.shape}, {residual.shape}")
        residual = self.residual_norm(residual)
        mean, logvar = self.distribution(condition)
        sq_mahal = (residual - mean).pow(2) * torch.exp(-logvar)
        return -0.5 * (sq_mahal + logvar).mean(dim=-1)

    def nll(self, condition: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return -self.forward(condition, residual)


class PatchScoreAggregator(nn.Module):
    def __init__(self, lower_tail_ratio: float = 0.2, alpha: float = 0.5, beta: float = 0.0):
        super().__init__()
        self.lower_tail_ratio = lower_tail_ratio
        self.alpha = alpha
        self.beta = beta

    def forward(self, patch_scores: torch.Tensor) -> torch.Tensor:
        mean_score = patch_scores.mean(dim=1)
        k = max(1, int(patch_scores.shape[1] * self.lower_tail_ratio))
        lower_tail = patch_scores.topk(k=k, dim=1, largest=False).values.mean(dim=1)
        std = patch_scores.std(dim=1)
        return mean_score + self.alpha * lower_tail - self.beta * std


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, lambd)


class OperatorHead(nn.Module):
    def __init__(self, input_dim: int, num_ops: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_ops),
        )

    def forward(self, features: torch.Tensor, grl_lambda: float = 1.0) -> torch.Tensor:
        pooled = features.mean(dim=1)
        pooled = gradient_reverse(pooled, grl_lambda)
        return self.net(pooled)

