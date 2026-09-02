# DBZ manga-panel LoRA evaluation

Generated 3 paired samples from the frozen base and trained adapter with identical prompts and seeds. The target profile comes from `validation-metadata.jsonl`.

| Metric | Held-out corpus | Base | Trained |
| --- | ---: | ---: | ---: |
| monochrome_score | 1.0000 | 0.7492 | 0.9342 |
| black_ink_ratio | 0.0717 | 0.4050 | 0.3590 |
| white_paper_ratio | 0.8147 | 0.1675 | 0.2326 |
| contrast | 0.4597 | 0.6665 | 0.7097 |
| edge_density | 0.1186 | 0.2651 | 0.2819 |

Mean normalized style distance (lower is closer): base **2.637**, trained **2.106**.

Adapter change: **20.1%** closer to the held-out corpus profile.

This is a small local theory test. The paired metrics measure movement toward monochrome ink/screentone statistics; they do not prove narrative coherence or exact character fidelity.
