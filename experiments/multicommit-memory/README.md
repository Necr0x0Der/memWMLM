# Multi-commit memory experiment

Date: 2026-11-25
Branch: `multicommit-memory`; pushed to `origin`; PR #17 open against `main`
PR: #17 (open), base `main`; no separate issue was created

## Claim

A frozen Qwen3-1.7B backbone with a position-aligned, writable memory sidecar
can answer questions supported by *earlier* commits, not only the last commit,
on a synthetic versioned-location task. The claim is deliberately narrow: one
entity, ordered commits, candidate-likelihood scoring.

## Protocol

For each episode:

1. source documents are separate commits (`Document 1`, `Document 2`, ...);
2. each document is processed by the frozen Qwen source path and written into
   position-aligned memory cells;
3. the application prompt contains only the question and a fixed instruction;
4. the application cannot see the source tokens; it reads memory through the
   trained reader and gated early-layer injection;
5. only application answer loss is supervised;
6. the answer is scored by candidate likelihood over the finite city vocabulary.

The base Qwen weights are frozen. Source-side and application-side rank-8 LoRA
adapters are trained, plus the writer/reader sidecar.

## Files

- `generate.py` — original complex multi-person dataset generator.
- `generate_simple.py` — single-entity, three-question diagnostic generator.
- `generate_two.py` — two-entity boundary generator.
- `common.py` — rendering, batching, memory writing, candidate scoring.
- `train.py` — reset-pair training; supports gradient clipping and gate init.
- `evaluate.py` — memory, null, shuffled, last-commit-only, full-context,
  application-only, and counterfactual controls.

## Quick commands

Run from the repository root with the Python environment that has the required
`torch`/`transformers`:

```text
PYTHONPATH=src python \
  experiments/multicommit-memory/generate_simple.py \
  --output-dir datasets/multicommit-simple3 \
  --train-episodes 2000 --test-episodes 400

PYTHONPATH=src python \
  experiments/multicommit-memory/train.py \
  --train-file datasets/multicommit-simple3/train.jsonl \
  --model-dir artifacts/models/Qwen3-1.7B \
  --output runs/multicommit-memory/simple3.pt \
  --steps 1500 --batch-size 8 --learning-rate 1e-4 \
  --source-lora-rank 8 --lora-rank 8 --gate-bias-init -2.0 \
  --max-grad-norm 1.0 --save-every 500 --log-every 250

PYTHONPATH=src python \
  experiments/multicommit-memory/evaluate.py \
  --checkpoint runs/multicommit-memory/simple3.pt \
  --test-file datasets/multicommit-simple3/test.jsonl \
  --train-file datasets/multicommit-simple3/train.jsonl \
  --split test --max-examples 1200 \
  --output runs/multicommit-memory/simple3_test.json
```

## Main result

`simple3.pt`, step 1500, 1200 test examples with unseen entity names:

| Variant | Accuracy |
|---|---:|
| `memory` | **99.75%** |
| `full_context` | 86.08% |
| `last_commit_only` | 50.25% |
| `shuffled_memory` | 72.33% |
| `null_memory` | 16.83% |
| `no_source` | 17.33% |
| counterfactual | follow 99.83%, resist 0.0% |

The discriminator is the `original` question: `last_commit_only` answers it
3.25% of the time, while full memory answers it 99.5%. `previous` is 100% under
memory but 58.25% under full context. This supports the narrow multi-commit
claim for the single-entity setting.

## Boundary

The complex three-person, 12-city task did not learn causal multi-commit
retrieval: memory was 9–11%, close to last-commit-only and shuffled controls.
The two-entity probe is undertrained at 1500 steps (15.25%) and reaches 53.75%
memory accuracy at 3000 steps while full context reaches 81.45%; `current` and
`previous` remain low. The single-entity positive result is therefore a narrow
claim, not evidence of general multi-entity binding. See `RESULTS.md` and
`RESEARCH-JOURNAL.md` for full metrics and caveats.
