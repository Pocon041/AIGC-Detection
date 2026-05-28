from __future__ import annotations

import io
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224, train: bool = False) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_plain_tensor_transform(image_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def denormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(max(3, round(sigma * 6) | 1))
    kernel = gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


def jpeg_compress_pil(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_tensor_perturbation(x_norm: torch.Tensor, mode: str) -> torch.Tensor:
    x = denormalize_tensor(x_norm)
    if mode.startswith("noise_"):
        sigma = float(mode.split("_")[1])
        x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
    elif mode.startswith("blur_"):
        sigma = float(mode.split("_")[1])
        x = gaussian_blur_tensor(x, sigma).clamp(0.0, 1.0)
    elif mode.startswith("resize_"):
        scale = float(mode.split("_")[1])
        h, w = x.shape[-2:]
        small = F.interpolate(x, size=(max(1, int(h * scale)), max(1, int(w * scale))), mode="bicubic", align_corners=False)
        x = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    elif mode == "identity":
        pass
    else:
        raise ValueError(f"鏈煡鎵板姩绫诲瀷: {mode}")
    return normalize_tensor(x)


def default_perturbation_modes() -> Dict[str, str]:
    return {
        "noise005": "noise_0.005",
        "noise010": "noise_0.010",
        "noise020": "noise_0.020",
        "blur05": "blur_0.5",
        "blur10": "blur_1.0",
        "resize075": "resize_0.75",
    }

