---
name: Experiment
about: Track one falsifiable experiment, its controls, artifacts, and result
title: "[experiment] "
labels: experiment
---

## Claim

One falsifiable claim. State what would confirm or refute it.

## Experiment

- Task / protocol:
- Model / backbone:
- Memory variant:
- Prompt version:
- Split:

## Data and config

- Dataset and hashes/manifest:
- Config path:
- Seed:
- Checkpoint path / step:

## Controls and baselines

- Frozen full-context:
- Application-only:
- `null_memory`:
- `shuffled_memory`:
- Novel-instance split:
- Counterfactual source swap:
- Matched exposure budget:

## Expected result

What signal is expected? What would be a negative or partial result?

## Artifacts

- `runs/...`
- `configs/...`
- `datasets/...`
- `RESULTS.md`

## Result

Metrics, train/test, controls, and qualitative failures.

## Caveats and open questions

- Distribution shift:
- Overfitting / train-test gap:
- Parameter or budget mismatch:
- Prompt/data provenance:
- What remains open:
