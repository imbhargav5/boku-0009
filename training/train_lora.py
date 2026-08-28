#!/usr/bin/env python3
"""Small, Apple-silicon-friendly Stable Diffusion LoRA trainer.

This intentionally trains only UNet attention LoRA weights. The base model,
VAE, and text encoder stay frozen, so the resulting artifact is a portable
character adapter rather than a new foundation model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


DEFAULT_BASE = "hakurei/waifu-diffusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--dataset", default="artifacts/model/dataset")
    parser.add_argument("--output", default="artifacts/model/goku-manga-lora")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class CaptionDataset(Dataset):
    def __init__(self, root: Path, resolution: int):
        self.root = root
        metadata_path = root / "metadata.jsonl"
        self.rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows found in {metadata_path}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["file_name"]).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "caption": row["text"]}


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "captions": [str(item["caption"]) for item in batch],
    }


def save_adapter(unet: UNet2DConditionModel, output: Path, base_model: str, step: int, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(unet)
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    # Diffusers' key conversion makes this loadable with pipe.load_lora_weights.
    state = convert_state_dict_to_diffusers(state)
    StableDiffusionPipeline.save_lora_weights(
        output,
        unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors",
        safe_serialization=True,
    )
    (output / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "target": "unet",
                "rank": args.rank,
                "lora_alpha": args.rank,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                "step": step,
                "resolution": args.resolution,
                "trigger_token": "goku_manga",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {output / 'pytorch_lora_weights.safetensors'} at step {step}", flush=True)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    dtype = torch.float32  # MPS does not reliably support bf16 training.
    print(f"device={device} base_model={args.base_model} steps={args.steps}", flush=True)

    dataset = CaptionDataset(Path(args.dataset), args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    text_encoder.eval()
    vae.eval()
    unet.train()
    for module in (text_encoder, vae):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet.add_adapter(lora_config)
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("UNet did not expose trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-config.json").write_text(
        json.dumps({**vars(args), "device": str(device), "dataset_size": len(dataset), "base_model": args.base_model}, indent=2)
        + "\n"
    )

    step = 0
    epoch = 0
    started = time.time()
    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            input_ids = tokenizer(
                batch["captions"],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 10 == 0 or step == args.steps:
                elapsed = max(time.time() - started, 1e-6)
                print(f"step={step}/{args.steps} loss={loss.item():.4f} steps_per_min={step / elapsed * 60:.2f}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_adapter(unet, output, args.base_model, step, args)

    (output / "training-complete.json").write_text(
        json.dumps({"steps": step, "epochs": epoch, "elapsed_seconds": time.time() - started}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
