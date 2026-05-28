from __future__ import annotations

import io
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CROP_SIZE = 224
DEFAULT_RESIZE_SIZE = 256


def default_resize_size(image_size: int = DEFAULT_CROP_SIZE) -> int:
    return int(round(int(image_size) * DEFAULT_RESIZE_SIZE / DEFAULT_CROP_SIZE))


def transform_protocol_name(image_size: int = DEFAULT_CROP_SIZE, resize_size: int | None = None) -> str:
    resize = default_resize_size(image_size) if resize_size is None else int(resize_size)
    return f"resize_short{resize}_crop{int(image_size)}"


def build_image_transform(image_size: int = DEFAULT_CROP_SIZE, train: bool = False, resize_size: int | None = None) -> Callable:
    resize = default_resize_size(image_size) if resize_size is None else int(resize_size)
    if train:
        return transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_plain_tensor_transform(image_size: int = DEFAULT_CROP_SIZE, train: bool = False, resize_size: int | None = None) -> Callable:
    resize = default_resize_size(image_size) if resize_size is None else int(resize_size)
    crop = transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        crop,
        transforms.ToTensor(),
    ])


class SynchronizedImageTransform:
    def __init__(
        self,
        image_size: int = DEFAULT_CROP_SIZE,
        train: bool = False,
        normalize: bool = True,
        resize_size: int | None = None,
    ):
        self.image_size = int(image_size)
        self.resize_size = default_resize_size(image_size) if resize_size is None else int(resize_size)
        self.train = bool(train)
        self.normalize = bool(normalize)

    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        if not images:
            return []
        resized = [
            TF.resize(image.convert("RGB"), self.resize_size, interpolation=transforms.InterpolationMode.BICUBIC)
            for image in images
        ]
        if self.train:
            min_width = min(image.width for image in resized)
            min_height = min(image.height for image in resized)
            if min_width < self.image_size or min_height < self.image_size:
                raise ValueError(f"resized image is smaller than crop size: min={(min_width, min_height)}, crop={self.image_size}")
            top = int(torch.randint(0, min_height - self.image_size + 1, (1,)).item())
            left = int(torch.randint(0, min_width - self.image_size + 1, (1,)).item())
            cropped = [TF.crop(image, top, left, self.image_size, self.image_size) for image in resized]
            if bool(torch.rand(()) < 0.5):
                cropped = [TF.hflip(image) for image in cropped]
        else:
            cropped = [TF.center_crop(image, [self.image_size, self.image_size]) for image in resized]
        tensors = [TF.to_tensor(image) for image in cropped]
        if self.normalize:
            tensors = [TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD) for tensor in tensors]
        return tensors


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
        raise ValueError(f"unknown perturbation mode: {mode}")
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

