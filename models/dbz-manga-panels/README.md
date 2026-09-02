# DBZ manga-panel LoRA

This subproject trains a model for a task that is deliberately separate from the repository's Goku character adapter: generating black-and-white Dragon Ball Z manga panels and short comic layouts.

## What the pipeline does

1. Collects a reproducible, provenance-preserving set of monochrome DBZ comic images from Safebooru's public JSON API.
2. Splits elongated comic pages into at most four non-overlapping page segments, normalizes them to square grayscale images, filters blank/low-information crops, and assigns train/validation by source image.
3. Trains only a rank-16 UNet LoRA over the frozen `hakurei/waifu-diffusion` base. Its trigger is `dbz_panel`; it does not overwrite or depend on the existing `goku_manga` adapter.
4. Generates identical-prompt/seed samples from the frozen base and trained adapter, then compares both against the held-out validation corpus's monochrome, ink, paper, contrast, and edge-density profile.
5. Asks the installed `qwen2.5vl:3b` Ollama model for a secondary rubric-based visual review of the paired sheet, then writes a final demo generation.

## Run end to end

From the repository root, after the existing `.venv` has the training requirements installed:

```bash
npm run manga-panels:e2e
```

The stages can also run independently:

```bash
npm run manga-panels:collect
npm run manga-panels:prepare
npm run manga-panels:train
npm run manga-panels:evaluate
npm run manga-panels:judge
npm run manga-panels:generate -- --prompt "dbz_panel, Dragon Ball Z, Vegeta close-up, black and white manga panel, bold ink, screentone"
```

Generated data and model binaries remain local and isolated here:

```text
models/dbz-manga-panels/
  data/raw/                         # downloaded sources + provenance manifest
  data/processed/                   # panel crops + captions + validation split
  artifacts/dbz-panel-lora/         # separate LoRA weights and training metrics
  artifacts/evaluation/             # paired samples, comparison image, report
  artifacts/generations/            # ad-hoc generations
```

The collector records source pages, direct URLs, SHA-256 hashes, tags, and dimensions. Those generated directories are gitignored; rerun the collector to reproduce the experiment locally.

## Verified local run

The pipeline was run on an Apple M1 Max with 32 GB unified memory:

- 48 cleaned source comics produced 111 training crops and 15 held-out validation crops.
- The rank-16 adapter trained for 300 steps (3 epochs) in 118.7 seconds on MPS and contains 3,319,808 trainable parameters. The resulting safetensors file is about 13 MB.
- On three identical-prompt/seed A/B generations, normalized distance to the held-out monochrome-panel profile fell from 2.637 for the base to 2.106 for the adapter: a 20.1% improvement.
- The local `qwen2.5vl:3b` judge preferred the trained image in all three A/B rows. This is secondary heuristic evidence, not ground truth.
- [`artifacts/final-demo.png`](artifacts/final-demo.png) is the final 18-step smoke generation, and [`artifacts/evaluation/comparison.png`](artifacts/evaluation/comparison.png) is the paired visual check.

## Scope of the result

This is a small local LoRA experiment, not a foundation model trained from scratch. It tests whether a compact adapter can move a known anime base toward monochrome panel composition. The statistical evaluation verifies that movement; visual review is still needed for anatomy, narrative continuity, readable lettering, and exact character identity.
