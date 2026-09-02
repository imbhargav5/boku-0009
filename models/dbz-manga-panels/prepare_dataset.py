#!/usr/bin/env python3
"""Turn long monochrome DBZ comic images into square panel-training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = MODEL_ROOT / "data/raw"
DEFAULT_OUTPUT = MODEL_ROOT / "data/processed"
CHARACTERS = {
    "son_goku": "Goku",
    "vegeta": "Vegeta",
    "piccolo": "Piccolo",
    "gohan": "Gohan",
    "son_gohan": "Gohan",
    "trunks": "Trunks",
    "bulma": "Bulma",
    "krillin": "Krillin",
    "freeza": "Frieza",
    "frieza": "Frieza",
    "cell_(dragon_ball)": "Cell",
    "majin_buu": "Majin Buu",
    "android_18": "Android 18",
    "android_17": "Android 17",
}
ACTION_TAGS = {
    "fighting": "martial arts fight",
    "punch": "punch",
    "kicking": "kick",
    "energy_ball": "energy attack",
    "kamehameha": "energy attack",
    "aura": "power aura",
    "angry": "angry expression",
    "injury": "battle damage",
    "blood": "battle damage",
    "speed_lines": "speed lines",
    "motion_lines": "motion lines",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-crops", type=int, default=4)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--trigger-token", default="dbz_panel")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def segment_boxes(width: int, height: int, max_crops: int) -> list[tuple[int, int, int, int]]:
    """Cover an elongated page with a bounded number of non-overlapping segments."""
    ratio = max(width, height) / max(min(width, height), 1)
    if ratio < 1.35:
        return [(0, 0, width, height)]
    count = min(max_crops, max(2, math.ceil(ratio)))
    boxes = []
    if height >= width:
        for index in range(count):
            top = round(index * height / count)
            bottom = round((index + 1) * height / count)
            boxes.append((0, top, width, bottom))
    else:
        for index in range(count):
            left = round(index * width / count)
            right = round((index + 1) * width / count)
            boxes.append((left, 0, right, height))
    return boxes


def square_grayscale(image: Image.Image, size: int) -> Image.Image:
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=0.5)
    grayscale.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    left = (size - grayscale.width) // 2
    top = (size - grayscale.height) // 2
    canvas.paste(grayscale, (left, top))
    return canvas.convert("RGB")


def visual_stats(image: Image.Image) -> dict[str, float]:
    values = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "ink_ratio": float((values < 210).mean()),
        "black_ratio": float((values < 80).mean()),
        "white_ratio": float((values > 245).mean()),
    }


def useful_crop(stats: dict[str, float]) -> bool:
    return stats["standard_deviation"] >= 18 and 0.025 <= stats["ink_ratio"] <= 0.72


def split_for(source_id: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def caption_for(record: dict, crop_count: int, trigger_token: str) -> str:
    tags = set(record.get("tags", []))
    characters = []
    for tag, name in CHARACTERS.items():
        if tag in tags and name not in characters:
            characters.append(name)
    actions = []
    for tag, phrase in ACTION_TAGS.items():
        if tag in tags and phrase not in actions:
            actions.append(phrase)
    layout = "four-panel comic layout" if "4koma" in tags else "manga page segment" if crop_count > 1 else "single manga panel"
    details = [*characters[:3], *actions[:3]]
    if "speech_bubble" in tags:
        details.append("speech bubbles")
    return ", ".join([
        trigger_token,
        "Dragon Ball Z",
        layout,
        "black and white manga",
        "bold ink line art",
        "screentone shading",
        "dynamic panel composition",
        *details,
    ])


def main() -> None:
    args = parse_args()
    if args.size < 128 or args.size % 8:
        raise ValueError("--size must be at least 128 and divisible by 8")
    manifest_path = args.raw / "source-manifest.jsonl"
    records = load_jsonl(manifest_path)
    if not records:
        raise ValueError(f"No source records in {manifest_path}")
    if args.output.exists():
        resolved_output = args.output.resolve()
        if resolved_output != DEFAULT_OUTPUT.resolve() and not (args.output / "dataset-summary.json").exists():
            raise ValueError(f"Refusing to replace unrecognized output directory: {resolved_output}")
        shutil.rmtree(args.output)
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True)
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    rejected: list[dict] = []
    seen_hashes: set[str] = set()
    source_split_counts = {"train": 0, "validation": 0}

    for record in records:
        source_path = args.raw / record["file"]
        with Image.open(source_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            boxes = segment_boxes(oriented.width, oriented.height, args.max_crops)
            split = split_for(str(record["id"]), args.validation_percent)
            source_split_counts[split] += 1
            for crop_index, box in enumerate(boxes, start=1):
                prepared = square_grayscale(oriented.crop(box), args.size)
                stats = visual_stats(prepared)
                if not useful_crop(stats):
                    rejected.append({"source_id": record["id"], "crop_index": crop_index, "reason": "low-information crop", "stats": stats})
                    continue
                digest = hashlib.sha256(prepared.tobytes()).hexdigest()
                if digest in seen_hashes:
                    rejected.append({"source_id": record["id"], "crop_index": crop_index, "reason": "duplicate crop"})
                    continue
                seen_hashes.add(digest)
                filename = f"{record['id']}-{crop_index:02d}.png"
                prepared.save(images_dir / filename, optimize=True)
                row = {
                    "file_name": f"images/{filename}",
                    "text": caption_for(record, len(boxes), args.trigger_token),
                    "source_id": str(record["id"]),
                    "source_file": record["file"],
                    "crop_index": crop_index,
                    "crop_count": len(boxes),
                    "crop_box": box,
                    "source_url": record["source_url"],
                    "stats": stats,
                }
                (train_rows if split == "train" else validation_rows).append(row)
                (images_dir / f"{record['id']}-{crop_index:02d}.txt").write_text(row["text"] + "\n")

    if len(train_rows) < 12:
        raise RuntimeError(f"Only {len(train_rows)} usable training crops were produced")
    (args.output / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    (args.output / "validation-metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in validation_rows))
    summary = {
        "source_manifest": str(manifest_path),
        "source_images": len(records),
        "source_splits": source_split_counts,
        "train_crops": len(train_rows),
        "validation_crops": len(validation_rows),
        "rejected_crops": len(rejected),
        "prepared_size": args.size,
        "max_crops_per_image": args.max_crops,
        "trigger_token": args.trigger_token,
        "rejected": rejected,
    }
    (args.output / "dataset-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(train_rows)} training and {len(validation_rows)} validation crops in {args.output}")


if __name__ == "__main__":
    main()
