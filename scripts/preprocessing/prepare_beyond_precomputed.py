from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from okk.dataset import read_manifest
from okk.transforms import build_plain_tensor_transform, transform_protocol_name
from okk.utils import configure_torch, get_device, list_images, set_seed


def require_diffusers():
    try:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit("需要安装 diffusers 和 transformers 才能生成论文版 diffusion-denoised variants") from exc
    return AutoencoderKL, DDIMScheduler, UNet2DConditionModel, CLIPTextModel, CLIPTokenizer


def pil_to_tensor(image: Image.Image, image_size: int):
    return build_plain_tensor_transform(image_size=image_size, train=False)(image.convert("RGB"))


def tensor_to_pil(x: torch.Tensor):
    x = x.detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(x)


def normalize_latents(latents: torch.Tensor, vae):
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    return latents * scale


def denormalize_latents(latents: torch.Tensor, vae):
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    return latents / scale


@torch.no_grad()
def encode_image(vae, image: torch.Tensor):
    x = image.unsqueeze(0) * 2.0 - 1.0
    posterior = vae.encode(x).latent_dist
    return normalize_latents(posterior.mode(), vae)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor):
    image = vae.decode(denormalize_latents(latents, vae)).sample
    return ((image.squeeze(0) + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def make_empty_prompt_embedding(tokenizer, text_encoder, device):
    tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


@torch.no_grad()
def ddim_denoise(unet, scheduler, latents: torch.Tensor, noise_level: int, num_steps: int, prompt_embedding: torch.Tensor, device):
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    if noise_level < 1 or noise_level >= len(timesteps):
        raise ValueError(f"noise_level must be in [1, {len(timesteps) - 1}]")
    start_index = len(timesteps) - noise_level
    start_t = timesteps[start_index]
    noise = torch.randn_like(latents)
    noisy = scheduler.add_noise(latents, noise, start_t.reshape(1))
    current = noisy
    for t in timesteps[start_index:]:
        noise_pred = unet(current, t, encoder_hidden_states=prompt_embedding).sample
        current = scheduler.step(noise_pred, t, current, eta=0.0).prev_sample
    return current


def collect_real_paths(manifest: str, split: str | None, real_dir: str):
    if manifest:
        items = read_manifest(manifest, split=split)
        return [Path(item.path) for item in items if item.label == 0]
    return sorted(list_images(Path(real_dir)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--real-dir", type=str, default="")
    parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-manifest", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--noise-levels", type=str, default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.manifest and not args.real_dir:
        raise ValueError("需要提供 --manifest 或 --real-dir")

    set_seed(args.seed)
    configure_torch(True)
    device = get_device(args.device)
    AutoencoderKL, DDIMScheduler, UNet2DConditionModel, CLIPTextModel, CLIPTokenizer = require_diffusers()

    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device)
    scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    vae.eval()
    unet.eval()
    text_encoder.eval()
    prompt_embedding = make_empty_prompt_embedding(tokenizer, text_encoder, device)

    paths = collect_real_paths(args.manifest, args.split if args.manifest else None, args.real_dir)
    if args.max_images > 0:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError("没有找到真实图像")

    levels = [int(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    original_dir = out_dir / "original"
    vae_dir = out_dir / "vae_recon"
    denoise_dirs = {level: out_dir / f"denoise_t{level}" for level in levels}
    original_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    for d in denoise_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in tqdm(paths, desc="prepare_beyond_variants"):
        stem = path.stem
        suffix = ".png"
        image = pil_to_tensor(Image.open(path), args.image_size).to(device)
        latents = encode_image(vae, image)
        original_out = original_dir / f"{stem}{suffix}"
        vae_out = vae_dir / f"{stem}{suffix}"
        tensor_to_pil(image).save(original_out)
        tensor_to_pil(decode_latents(vae, latents)).save(vae_out)
        row = {"id": stem, "split": args.split, "original": str(original_out), "vae_recon": str(vae_out)}
        for level in levels:
            denoised_latents = ddim_denoise(unet, scheduler, latents, level, args.num_steps, prompt_embedding, device)
            denoised = decode_latents(vae, denoised_latents)
            out_path = denoise_dirs[level] / f"{stem}{suffix}"
            tensor_to_pil(denoised).save(out_path)
            row[f"denoise_t{level}"] = str(out_path)
        rows.append(row)

    fieldnames = ["id", "split", "original", "vae_recon"] + [f"denoise_t{level}" for level in levels]
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote precomputed manifest: {out_manifest}, samples: {len(rows)}, transform={transform_protocol_name(args.image_size)}")


if __name__ == "__main__":
    main()


