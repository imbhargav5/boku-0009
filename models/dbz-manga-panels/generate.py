#!/usr/bin/env python3
"""Generate a manga panel with the trained dbz_panel LoRA."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = "dbz_panel, Dragon Ball Z, Goku and Vegeta martial arts duel, black and white manga panel, bold ink line art, screentone shading, speed lines, dramatic low angle, speech bubbles"
DEFAULT_NEGATIVE = "color, painting, 3d render, photorealistic, blurry, low contrast, watermark, logo"


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODEL_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="hakurei/waifu-diffusion")
    parser.add_argument("--lora", type=Path, default=MODEL_ROOT / "artifacts/dbz-panel-lora")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float32, safety_checker=None)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    image = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]
    if args.output:
        output = args.output
    else:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S-%f%z")
        output = MODEL_ROOT / "artifacts/generations" / stamp / "dbz-panel.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    metadata = {
        "base_model": args.base_model,
        "lora": portable_path(args.lora),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance": args.guidance,
        "width": args.width,
        "height": args.height,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
