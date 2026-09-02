#!/usr/bin/env python3
"""Generate paired base/LoRA samples and compare their manga-style statistics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageOps


MODEL_ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "Dragon Ball Z, Goku and Vegeta martial arts duel, black and white manga panel, dramatic speed lines, bold ink",
    "Dragon Ball Z, Piccolo fires an energy blast, black and white manga panel, screentone shadows, low angle",
    "Dragon Ball Z, Gohan determined close-up, cracked battlefield, black and white manga panel, speech bubble",
]
NEGATIVE_PROMPT = "color, painting, 3d render, photorealistic, blurry, low contrast, watermark, logo"


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODEL_ROOT))
    except ValueError:
        return str(path)


def image_features(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray_image = ImageOps.grayscale(image)
    gray = np.asarray(gray_image, dtype=np.float32)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    edges = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {
        "monochrome_score": float(1.0 - np.mean(chroma) / 255.0),
        "black_ink_ratio": float(np.mean(gray < 80)),
        "white_paper_ratio": float(np.mean(gray > 225)),
        "contrast": float(min(np.std(gray) / 127.5, 1.0)),
        "edge_density": float(np.mean(edges > 42)),
    }


def mean_features(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def style_distance(features: dict[str, float], target: dict[str, float]) -> float:
    scales = {
        "monochrome_score": 0.08,
        "black_ink_ratio": 0.10,
        "white_paper_ratio": 0.18,
        "contrast": 0.16,
        "edge_density": 0.08,
    }
    return float(np.mean([abs(features[key] - target[key]) / scales[key] for key in target]))


def dataset_profile(dataset: Path) -> tuple[dict[str, float], str]:
    validation_path = dataset / "validation-metadata.jsonl"
    metadata_path = validation_path if validation_path.exists() and validation_path.read_text().strip() else dataset / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    features = [image_features(Image.open(dataset / row["file_name"])) for row in rows]
    return mean_features(features), metadata_path.name


def contact_sheet(pairs: list[dict], output: Path) -> None:
    tile = 384
    label_height = 28
    sheet = Image.new("RGB", (tile * 2, (tile + label_height) * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, pair in enumerate(pairs):
        top = row_index * (tile + label_height)
        for column, kind in enumerate(("baseline", "trained")):
            image = Image.open(pair[f"{kind}_path"]).convert("RGB").resize((tile, tile))
            sheet.paste(image, (column * tile, top))
            label = f"{kind} | seed {pair['seed']} | prompt {row_index + 1}"
            draw.text((column * tile + 8, top + tile + 7), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def markdown_report(report: dict) -> str:
    lines = [
        "# DBZ manga-panel LoRA evaluation",
        "",
        f"Generated {len(report['pairs'])} paired samples from the frozen base and trained adapter with identical prompts and seeds. The target profile comes from `{report['reference_profile']}`.",
        "",
        "| Metric | Held-out corpus | Base | Trained |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in report["dataset_profile"]:
        lines.append(f"| {key} | {report['dataset_profile'][key]:.4f} | {report['baseline_mean'][key]:.4f} | {report['trained_mean'][key]:.4f} |")
    lines.extend([
        "",
        f"Mean normalized style distance (lower is closer): base **{report['baseline_style_distance']:.3f}**, trained **{report['trained_style_distance']:.3f}**.",
        "",
        f"Adapter change: **{report['style_distance_improvement_percent']:.1f}%** closer to the held-out corpus profile.",
        "",
        "This is a small local theory test. The paired metrics measure movement toward monochrome ink/screentone statistics; they do not prove narrative coherence or exact character fidelity.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="hakurei/waifu-diffusion")
    parser.add_argument("--lora", type=Path, default=MODEL_ROOT / "artifacts/dbz-panel-lora")
    parser.add_argument("--dataset", type=Path, default=MODEL_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "artifacts/evaluation")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    profile, reference_profile = dataset_profile(args.dataset)
    started = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float32, safety_checker=None)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pairs = []
    for index, prompt in enumerate(PROMPTS):
        seed = args.seed + index
        common = {
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "width": args.width,
            "height": args.height,
        }
        baseline = pipe(prompt, generator=torch.Generator(device="cpu").manual_seed(seed), **common).images[0]
        baseline_path = args.output / "baseline" / f"sample-{index + 1:02d}.png"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline.save(baseline_path)
        pairs.append({"prompt": prompt, "seed": seed, "baseline_path": portable_path(baseline_path), "baseline_features": image_features(baseline)})
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    for index, pair in enumerate(pairs):
        prompt = f"dbz_panel, {pair['prompt']}"
        common = {
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "width": args.width,
            "height": args.height,
        }
        trained = pipe(prompt, generator=torch.Generator(device="cpu").manual_seed(pair["seed"]), **common).images[0]
        trained_path = args.output / "trained" / f"sample-{index + 1:02d}.png"
        trained_path.parent.mkdir(parents=True, exist_ok=True)
        trained.save(trained_path)
        pair.update({"trained_prompt": prompt, "trained_path": portable_path(trained_path), "trained_features": image_features(trained)})

    baseline_mean = mean_features([pair["baseline_features"] for pair in pairs])
    trained_mean = mean_features([pair["trained_features"] for pair in pairs])
    baseline_distance = style_distance(baseline_mean, profile)
    trained_distance = style_distance(trained_mean, profile)
    improvement = (baseline_distance - trained_distance) / max(baseline_distance, 1e-8) * 100
    report = {
        "base_model": args.base_model,
        "lora": portable_path(args.lora),
        "device": str(device),
        "generation_steps": args.steps,
        "reference_profile": reference_profile,
        "dataset_profile": profile,
        "baseline_mean": baseline_mean,
        "trained_mean": trained_mean,
        "baseline_style_distance": baseline_distance,
        "trained_style_distance": trained_distance,
        "style_distance_improvement_percent": improvement,
        "elapsed_seconds": time.time() - started,
        "pairs": pairs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "report.md").write_text(markdown_report(report))
    contact_sheet(pairs, args.output / "comparison.png")
    print(json.dumps({
        "baseline_style_distance": baseline_distance,
        "trained_style_distance": trained_distance,
        "style_distance_improvement_percent": improvement,
        "report": str(args.output / "report.md"),
        "comparison": str(args.output / "comparison.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
