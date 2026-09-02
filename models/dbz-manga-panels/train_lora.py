#!/usr/bin/env python3
"""Train the isolated DBZ manga-panel UNet LoRA on Apple Silicon or CUDA."""

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


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = "hakurei/waifu-diffusion"


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODEL_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--dataset", type=Path, default=MODEL_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "artifacts/dbz-panel-lora")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trigger-token", default="dbz_panel")
    return parser.parse_args()


class CaptionDataset(Dataset):
    def __init__(self, root: Path, resolution: int):
        self.root = root
        metadata_path = root / "metadata.jsonl"
        self.rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"No rows found in {metadata_path}")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(self.root / row["file_name"]).convert("RGB")
        return {"pixel_values": self.transform(image), "caption": row["text"]}


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "captions": [str(item["caption"]) for item in batch],
    }


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_adapter(unet: UNet2DConditionModel, output: Path, args: argparse.Namespace, step: int) -> None:
    state = get_peft_model_state_dict(unet)
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    state = convert_state_dict_to_diffusers(state)
    StableDiffusionPipeline.save_lora_weights(
        output,
        unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors",
        safe_serialization=True,
    )
    (output / "adapter_config.json").write_text(json.dumps({
        "base_model": args.base_model,
        "target": "unet",
        "rank": args.rank,
        "lora_alpha": args.rank,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "step": step,
        "resolution": args.resolution,
        "trigger_token": args.trigger_token,
    }, indent=2) + "\n")
    print(f"Saved adapter at step {step}: {output / 'pytorch_lora_weights.safetensors'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    dtype = torch.float32
    dataset = CaptionDataset(args.dataset, args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    print(f"device={device} base_model={args.base_model} examples={len(dataset)} steps={args.steps}", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    for module in (text_encoder, vae):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    unet.to(device).train()
    unet.add_adapter(LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    ))
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("UNet exposed no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "dataset": portable_path(args.dataset),
        "output": portable_path(args.output),
        "device": str(device),
        "dataset_size": len(dataset),
    }
    (args.output / "training-config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    metrics_path = args.output / "training-metrics.jsonl"
    metrics_path.write_text("")

    step = 0
    epoch = 0
    losses: list[float] = []
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
                hidden_states = text_encoder(input_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prediction = unet(noisy_latents, timesteps, hidden_states).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), noise.float(), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_value, "elapsed_seconds": time.time() - started}) + "\n")
            if step == 1 or step % 10 == 0 or step == args.steps:
                elapsed = max(time.time() - started, 1e-6)
                print(f"step={step}/{args.steps} loss={loss_value:.4f} steps_per_min={step / elapsed * 60:.2f}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_adapter(unet, args.output, args, step)

    elapsed = time.time() - started
    final_window = losses[-min(25, len(losses)):]
    summary = {
        "steps": step,
        "epochs": epoch,
        "elapsed_seconds": elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_25_step_mean_loss": float(np.mean(final_window)),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    (args.output / "training-complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
