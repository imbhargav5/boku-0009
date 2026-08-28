#!/usr/bin/env python3
"""Generate a PNG with the trained character adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="hakurei/waifu-diffusion")
    parser.add_argument("--lora", default="artifacts/model/goku-manga-lora")
    parser.add_argument(
        "--prompt",
        default="goku_manga, solo martial artist, spiky black hair, orange and blue gi, dynamic three-quarter pose, clean manga linework",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG path to overwrite explicitly, or a directory under which to create a timestamped run (default: artifacts/model/generations)",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, dtype=torch.float32, safety_checker=None)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]
    if args.output is None:
        output_root = Path("artifacts/model/generations")
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f%z")
        output = output_root / timestamp / "goku-manga.png"
    else:
        requested_output = Path(args.output)
        if requested_output.suffix.lower() == ".png":
            output = requested_output
        else:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f%z")
            output = requested_output / timestamp / "goku-manga.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    (output.with_suffix(".json")).write_text(
        json.dumps({"base_model": args.base_model, "lora": args.lora, "prompt": args.prompt, "seed": args.seed}, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
