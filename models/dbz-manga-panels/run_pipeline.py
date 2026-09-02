#!/usr/bin/env python3
"""Run collection, preprocessing, LoRA training, and paired evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODEL_ROOT.parents[1]


def run(label: str, command: list[str]) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    config = json.loads((MODEL_ROOT / "config.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--limit", type=int, default=config["raw_image_limit"])
    parser.add_argument("--steps", type=int, default=config["training_steps"])
    args = parser.parse_args()
    if not args.skip_collect:
        run("collect", ["node", str(MODEL_ROOT / "scripts/collect_dataset.mjs"), "--limit", str(args.limit)])
    run("prepare", [
        sys.executable,
        str(MODEL_ROOT / "prepare_dataset.py"),
        "--size", str(config["prepared_size"]),
        "--max-crops", str(config["max_crops_per_image"]),
        "--trigger-token", config["trigger_token"],
    ])
    if not args.skip_train:
        run("train", [
            sys.executable,
            str(MODEL_ROOT / "train_lora.py"),
            "--base-model", config["base_model"],
            "--resolution", str(config["training_resolution"]),
            "--steps", str(args.steps),
            "--learning-rate", str(config["learning_rate"]),
            "--rank", str(config["rank"]),
            "--seed", str(config["seed"]),
            "--trigger-token", config["trigger_token"],
        ])
    if not args.skip_evaluate:
        run("evaluate", [sys.executable, str(MODEL_ROOT / "evaluate.py")])
    if not args.skip_judge:
        run("judge", [sys.executable, str(MODEL_ROOT / "judge_samples.py")])
    if not args.skip_generate:
        run("generate", [
            sys.executable,
            str(MODEL_ROOT / "generate.py"),
            "--output", str(MODEL_ROOT / "artifacts/final-demo.png"),
            "--seed", str(config["seed"]),
        ])
    print(f"\nCompleted DBZ manga-panel pipeline. Outputs: {MODEL_ROOT / 'artifacts'}", flush=True)


if __name__ == "__main__":
    main()
