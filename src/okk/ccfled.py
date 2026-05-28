from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from okk.transforms import denormalize_tensor


STRUCTURAL_PROXY_NAMES = [
    "edge_mean",
    "edge_std",
    "edge_density",
    "spectral_entropy",
    "high_freq_ratio",
    "patch_var_entropy",
    "patch_var_mean",
    "color_std_mean",
]

PIPELINE_PROXY_NAMES = [
    "log_area",
    "aspect_ratio",
    "jpeg_quant_mean",
    "is_jpeg",
]

DEFAULT_CONDITION_PROXY_NAMES = [
    "edge_mean",
    "edge_density",
    "spectral_entropy",
    "patch_var_entropy",
    "patch_var_mean",
    "color_std_mean",
]


def proxy_names(include_pipeline: bool = True) -> list[str]:
    names = list(STRUCTURAL_PROXY_NAMES)
    if include_pipeline:
        names.extend(PIPELINE_PROXY_NAMES)
    return names


def pool_tokens(tokens: torch.Tensor, mode: str = "mean", topk_ratio: float = 0.2) -> torch.Tensor:
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "topk_l2":
        k = max(1, int(round(tokens.shape[1] * topk_ratio)))
        token_scores = tokens.pow(2).mean(dim=-1)
        idx = token_scores.topk(k=k, dim=1, largest=True).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return tokens.gather(1, gather_idx).mean(dim=1)
    raise ValueError(f"unknown token pooling mode: {mode}")


def compute_proxy_features(x_norm: torch.Tensor, paths: Sequence[str]) -> torch.Tensor:
    x01 = denormalize_tensor(x_norm).float()
    gray = (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]).clamp(0.0, 1.0)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x01.device,
        dtype=x01.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    edge_mean = grad.flatten(1).mean(dim=1)
    edge_std = grad.flatten(1).std(dim=1)
    edge_density = (grad > 0.08).float().flatten(1).mean(dim=1)

    spectrum = torch.fft.rfft2(gray.squeeze(1), norm="ortho").abs().pow(2)
    spectrum[:, 0, 0] = 0.0
    flat_power = spectrum.flatten(1)
    prob = flat_power / flat_power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    spectral_entropy = -(prob * (prob.clamp_min(1e-12).log())).sum(dim=1) / math.log(prob.shape[1])

    h, w_half = spectrum.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x01.device, dtype=x01.dtype).view(h, 1)
    fx = torch.fft.rfftfreq((w_half - 1) * 2, device=x01.device, dtype=x01.dtype).view(1, w_half)
    radius = torch.sqrt(fx.pow(2) + fy.pow(2))
    high_mask = radius > 0.35
    high_freq_ratio = spectrum[:, high_mask].sum(dim=1) / spectrum.flatten(1).sum(dim=1).clamp_min(1e-12)

    patch_size = max(8, gray.shape[-1] // 16)
    patches = F.unfold(gray, kernel_size=patch_size, stride=patch_size)
    patch_var = patches.var(dim=1, unbiased=False)
    patch_prob = patch_var / patch_var.sum(dim=1, keepdim=True).clamp_min(1e-12)
    patch_var_entropy = -(patch_prob * patch_prob.clamp_min(1e-12).log()).sum(dim=1) / math.log(patch_prob.shape[1])
    patch_var_mean = patch_var.mean(dim=1)

    color_std_mean = x01.flatten(2).std(dim=2).mean(dim=1)
    metadata = torch.tensor(
        [_image_metadata_features(path) for path in paths],
        device=x01.device,
        dtype=x01.dtype,
    )

    structural = torch.stack(
        [
            edge_mean,
            edge_std,
            edge_density,
            spectral_entropy,
            high_freq_ratio,
            patch_var_entropy,
            patch_var_mean,
            color_std_mean,
        ],
        dim=1,
    )
    return torch.cat([structural, metadata], dim=1)


def _image_metadata_features(path: str) -> list[float]:
    path_obj = Path(path)
    is_jpeg = 1.0 if path_obj.suffix.lower() in {".jpg", ".jpeg"} else 0.0
    try:
        with Image.open(path_obj) as image:
            width, height = image.size
            quant_mean = _jpeg_quant_mean(image)
    except Exception:
        width, height, quant_mean = 1, 1, -1.0
    area = max(1, int(width) * int(height))
    aspect = float(width) / max(1.0, float(height))
    return [math.log(float(area)), aspect, quant_mean, is_jpeg]


def _jpeg_quant_mean(image: Image.Image) -> float:
    quant = getattr(image, "quantization", None)
    if not quant:
        return -1.0
    values: list[float] = []
    for table in quant.values():
        values.extend(float(v) for v in table)
    if not values:
        return -1.0
    return float(np.mean(values))


def parse_name_list(value: str, available: Sequence[str], default: Sequence[str]) -> list[str]:
    if not value or value == "default":
        names = list(default)
    elif value == "all":
        names = list(available)
    else:
        names = [item.strip() for item in value.split(",") if item.strip()]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"unknown proxy columns {missing}; available={list(available)}")
    return names


def select_columns(values: np.ndarray, names: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    return values[:, [index[name] for name in selected]]


def standardize_from_bank(bank: np.ndarray, values: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = bank.mean(axis=0, keepdims=True)
    std = bank.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (values - mean) / std, mean.squeeze(0), std.squeeze(0)


def diagonal_global_energy(bank_z: np.ndarray, eval_z: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    mean = bank_z.mean(axis=0)
    var = bank_z.var(axis=0) + float(shrinkage)
    return ((eval_z - mean) ** 2 / np.maximum(var, 1e-8)).mean(axis=1)


def diagonal_local_energy(
    bank_z: np.ndarray,
    eval_z: np.ndarray,
    neighbor_idx: np.ndarray,
    shrinkage: float = 0.1,
    chunk_size: int = 2048,
) -> np.ndarray:
    global_var = bank_z.var(axis=0)
    scores = np.empty(eval_z.shape[0], dtype=np.float64)
    for start in range(0, eval_z.shape[0], chunk_size):
        end = min(start + chunk_size, eval_z.shape[0])
        local = bank_z[neighbor_idx[start:end]]
        mean = local.mean(axis=1)
        var = local.var(axis=1) + float(shrinkage) * global_var[None, :] + 1e-8
        scores[start:end] = ((eval_z[start:end] - mean) ** 2 / var).mean(axis=1)
    return scores


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def eta_squared(values: np.ndarray, groups: Iterable[str]) -> float:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(list(groups)).astype(str)
    mask = np.isfinite(values)
    values = values[mask]
    groups = groups[mask]
    if len(values) < 2:
        return float("nan")
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for group in np.unique(groups):
        group_values = values[groups == group]
        if len(group_values):
            between += len(group_values) * (group_values.mean() - grand) ** 2
    return float(between / total)


def string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)
