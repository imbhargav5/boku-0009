# Reference manifests

`reference-manifest.jsonl` is the first-pass training/evaluation manifest. It contains only the 76 manually retained references. `deferred-manifest.jsonl` records the 124 images moved into `references/goku/deferred/`.

The manifests are the source of truth for the next data-collection step. The original images and source metadata are preserved; nothing was deleted. `valid` images must remain held out from Artist improvement and SFT curation.

Each retained record has a prompt that tells the Artist what visual task to perform. The collector first sends a normalized reference image to the vision analyzer and saves a structured `reference-analysis.json`; that brief is then included in the Artist prompt. With `--artist-vision`, the Ollama Artist also receives the reference image directly. The reference is always supplied to the Judge during candidate scoring.
