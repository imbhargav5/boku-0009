#!/usr/bin/env python3
"""Ask the installed local vision model to compare the paired evaluation sheet."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
RUBRIC = """You are reviewing a three-row A/B comparison sheet for a small local image-model experiment.
In every row the LEFT image is the frozen anime base and the RIGHT image is the trained DBZ manga-panel LoRA. Labels under each row confirm this.

For each row, score left and right from 0 to 100 using this rubric:
- 25 points: convincingly black-and-white manga ink and screentone
- 20 points: readable panel composition and focal hierarchy
- 20 points: Dragon Ball Z visual vocabulary
- 20 points: anatomy and action clarity
- 15 points: useful clean output without painterly/color artifacts

Return one compact JSON object matching the supplied schema. Inspect the actual pixels and assign distinct evidence-based scores; do not use placeholder values. Keep the summary under 30 words. Do not reward an image merely for being monochrome if its anatomy or composition is incoherent."""
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer", "minimum": 1, "maximum": 3},
                    "baseline_score": {"type": "number", "minimum": 0, "maximum": 100},
                    "trained_score": {"type": "number", "minimum": 0, "maximum": 100},
                },
                "required": ["row", "baseline_score", "trained_score"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["pairs", "summary"],
}


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODEL_ROOT))
    except ValueError:
        return str(path)


def request_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_model_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_and_validate(result: dict) -> None:
    pairs = result.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError("Judge response must contain exactly three pair results")
    for index, pair in enumerate(pairs, start=1):
        try:
            pair["row"] = int(pair.get("row"))
            pair["baseline_score"] = float(pair.get("baseline_score"))
            pair["trained_score"] = float(pair.get("trained_score"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric fields for row {index}") from error
        if pair["row"] != index:
            raise ValueError(f"Judge row {index} is missing or out of order")
        for key in ("baseline_score", "trained_score"):
            value = pair.get(key)
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"Invalid {key} for row {index}")
        difference = pair["trained_score"] - pair["baseline_score"]
        winner = "trained" if difference > 0 else "baseline" if difference < 0 else "tie"
        pair["winner"] = winner
    wins = [pair["winner"] for pair in pairs]
    overall = "trained" if wins.count("trained") > wins.count("baseline") else "baseline" if wins.count("baseline") > wins.count("trained") else "tie"
    result["overall_winner"] = overall


def markdown(result: dict, model: str) -> str:
    lines = [
        "# Local vision-judge review",
        "",
        f"Model: `{model}`",
        "",
        "| Row | Base | Trained | Winner | Evidence |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for pair in result["pairs"]:
        reason = str(pair.get("reason", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {pair['row']} | {pair['baseline_score']} | {pair['trained_score']} | {pair['winner']} | {reason} |")
    lines.extend([
        "",
        f"Overall: **{result.get('overall_winner', 'unknown')}** — {result.get('summary', '')}",
        "",
        "Visible limitations:",
        "",
    ])
    lines.extend(f"- {item}" for item in result.get("limitations", []))
    lines.extend(["", "This is a secondary heuristic review by a small local VLM, not ground truth.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=MODEL_ROOT / "artifacts/evaluation/comparison.png")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "artifacts/evaluation")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    image_data = base64.b64encode(args.image.read_bytes()).decode()
    payload = {
        "model": args.model,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "messages": [{"role": "user", "content": RUBRIC, "images": [image_data]}],
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
    }
    try:
        response = request_json(f"{args.host.rstrip('/')}/api/chat", payload, args.timeout)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {args.host}: {error}") from error
    result = parse_model_json(response.get("message", {}).get("content", ""))
    normalize_and_validate(result)
    result["model"] = args.model
    result["comparison_image"] = portable_path(args.image)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vision-judge.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "vision-judge.md").write_text(markdown(result, args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
