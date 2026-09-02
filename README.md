# Local drawing improvement loop

This project implements the practical part of the shared method: a local **reference analysis → Artist → SVG → Chromium → PNG → Vision Judge → feedback → improved Artist** loop.

It deliberately starts with SVG instead of unrestricted Canvas JavaScript. SVG is easier for a coding model to edit, easier to validate, and gives the judge a stable vocabulary such as `hair`, `left-eye`, and `jaw`. The renderer treats every model response as untrusted input and only accepts a restricted, self-contained SVG subset.

## What is included

- Candidate branching: generate several drawings each round and retain the highest-scoring one.
- A secure-ish SVG gate: no scripts, event handlers, remote assets, embedded HTML, or SVG filters.
- Deterministic Chromium screenshots, written as PNG artifacts.
- A model-provider interface plus an OpenAI-compatible local provider for MLX, Ollama-compatible servers, or another local endpoint.
- An image-analysis pass that turns each reference into a structured visual brief for the text-only Artist.
- An optional image-conditioned Ollama Artist (`--artist-vision`) for models such as `qwen2.5vl:3b`.
- An offline fixture provider so the full render/select/improve loop can be run before configuring models.
- Durable per-round artifacts and `trajectory.jsonl`, plus a final local image-dataset export and LoRA training stage.

The implementation uses an *original* avatar in its fixture. Supply only reference images you have the right to use.

## Quick start

```bash
npm install
npx playwright install chromium
npm run demo
```

The demo creates a timestamped directory under `artifacts/`. It contains all candidate SVGs, their PNG renders, each score report, and the selected winner for every round.

## Use Ollama locally

This repository has a native Ollama mode. With the two recommended local models available, run:

```bash
npm run ollama -- \
  --reference ./references/original-character.png \
  --prompt "Draw an original front-facing, neutral-expression martial-artist avatar" \
  --candidates 4 \
  --rounds 3
```

It uses `qwen2.5-coder:latest` as the SVG Artist, `qwen2.5vl:3b` as the reference analyzer and Judge by default. For stronger image conditioning, pass `--artist-vision` to use `qwen2.5vl:3b` as the Artist too. Override model names with `ARTIST_MODEL`, `VISION_MODEL`, or `JUDGE_MODEL`, or point at a non-default local service with `OLLAMA_HOST`.
Live model requests have a 120-second deadline by default (`MODEL_REQUEST_TIMEOUT_MS`); a timed-out attempt is retried once and then recorded as an error for that candidate.

## Collect trajectories from a reference manifest

Step 3 uses the curated JSONL manifest at `data/reference-manifest.jsonl`. Each training record gets its own resumable artifact directory, normalized `reference.png`, candidate SVG/PNG files, score reports, and `trajectory.jsonl`. JPEG and PNG references are normalized through Chromium so the Judge always sees the same 768px canvas.

Run a small pilot first:

```bash
npm run collect -- \
  --max-images 3 \
  --candidates 2 \
  --rounds 1 \
  --artist-vision \
  --out artifacts/collection-pilot
```

Then run the full training split (the default is four candidates for two rounds):

```bash
npm run collect -- \
  --manifest data/reference-manifest.jsonl \
  --split train \
  --artist-vision \
  --candidates 4 \
  --rounds 2 \
  --out artifacts/collection-train
```

The collector writes `collection-index.jsonl` and `collection-summary.json`, plus a `reference-analysis.json` and `manifest-record.json` beside each reference's artifacts so later SFT/LoRA conversion can recover the visual brief, exact prompt, and source image. Re-running with the same output directory skips records that already have a `winner.json`; failed records remain indexed and can be retried after fixing the model or reference. Individual analysis/Artist/Judge failures are retried once and recorded as `*.error.json` without discarding the rest of a reference's candidates.

## Use another local model server

Start an OpenAI-compatible local model server for a coding model and a vision-language model. On an M-series Mac, run one model at a time; the loop does that naturally: Artist calls complete before the Judge calls begin.

Set the endpoints and model names in your environment:

```bash
export ARTIST_ENDPOINT=http://127.0.0.1:8000/v1
export ARTIST_MODEL=your-local-coding-model
export JUDGE_ENDPOINT=http://127.0.0.1:8001/v1
export JUDGE_MODEL=your-local-vision-model
```

Then run a job with a reference image:

```bash
npm run loop -- \
  --reference ./references/original-character.png \
  --prompt "Draw an original front-facing, neutral-expression martial-artist avatar" \
  --candidates 4 \
  --rounds 3
```

The Judge endpoint must support OpenAI-style image inputs. The Artist is asked to return only SVG. A local server that uses a slightly different API can be connected by adding a small provider in `src/providers.ts`; the orchestration layer is independent of the serving stack.

## Output contract

Each run creates:

```text
artifacts/<run-id>/
  config.json
  trajectory.jsonl
  round-01/
    candidate-01.svg
    candidate-01.png
    candidate-01.score.json
    winner.json
```

`trajectory.jsonl` captures the prompt, parent candidate, generated SVG, score, and critique for every attempt. Keep the best verified trajectories for later SFT/LoRA; do not blindly train on judge mistakes.

## Architecture

```text
reference image + prompt
            |
            v
      Vision analyzer  -- produces structured visual brief
            |
            v
      Artist provider  -- produces constrained SVG
            |
            v
      Chromium renderer -- produces candidate PNG
            |
            v
      Vision Judge      -- returns numeric rubric + critique
            |
            v
   retain best candidate, feed critique into next round
```

This is an evaluation-and-search system, not GRPO. It proves that the reward loop has signal and creates useful trajectories before any expensive training work.

## Train the character adapter locally

The final stage can now produce a real text-to-image model artifact. It uses the public `hakurei/waifu-diffusion` Stable-Diffusion-1.5-compatible anime base and trains only a small UNet LoRA adapter. The base model remains frozen; the output is a portable `pytorch_lora_weights.safetensors`, not a foundation model trained from scratch.

Create the isolated Python environment and install the Apple-silicon-friendly stack once:

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
uv pip install --python .venv/bin/python \
  torch torchvision diffusers transformers accelerate peft safetensors pillow huggingface_hub
```

Export the completed training split (the collector's normalized `reference.png` files plus captions):

```bash
npm run prepare:model-dataset
```

Train 200 steps on the laptop. This uses MPS when available and defaults to a conservative 384px resolution, batch size 1, and float32 because Apple MPS does not reliably train with bf16. The latest adapter is serialized every 50 steps:

```bash
npm run train:lora
```

Generate a sample with the adapter. With no `--output`, every run gets its own timestamped folder:

```bash
npm run generate:model
```

```text
artifacts/model/generations/2026-08-29T15-42-10-123456+0530/
  goku-manga.png
  goku-manga.json
```

Pass an explicit `.png` path only when you intentionally want to overwrite a particular file, for example `--output artifacts/model/goku-manga-sample.png`.

Generated datasets and timestamped runs stay under the ignored `artifacts/model/` directory. The final adapter metadata and one sample are versioned so the repository includes a concrete trained result:

```text
artifacts/model/
  dataset/metadata.jsonl
  goku-manga-lora/pytorch_lora_weights.safetensors
  goku-manga-lora/adapter_config.json
  generations/<timestamp>/goku-manga.png
```

The `goku_manga` token in the captions and default prompt is the adapter's trigger. Later, the same trainer can be rerun with more curated images, a higher step count, or a stronger SDXL/anime base on a larger GPU.

## Train the separate DBZ manga-panel model

The character adapter above is intentionally not reused for panel generation. A second, isolated pipeline lives in [`models/dbz-manga-panels`](models/dbz-manga-panels/README.md). It collects monochrome DBZ comic references, slices long pages into panel-sized training examples, trains a LoRA with the separate `dbz_panel` trigger, and produces a paired base-versus-trained evaluation.

```bash
npm run manga-panels:e2e
```

Its dataset, weights, evaluation report, and generated samples stay under that model's own subdirectory.
