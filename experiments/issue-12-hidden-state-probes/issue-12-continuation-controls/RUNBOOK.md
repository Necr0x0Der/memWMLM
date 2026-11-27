# Running on a GPU machine

Run all commands from the repository root. Requirements: the existing `.venv`
with PyTorch/Transformers, and the dataset at
`experiments/issue-12-hidden-state-probes/artifacts/db`, containing
`manifest.json`, `train.jsonl`, `validation.jsonl`, and `test.jsonl`.
The dataset is in an ignored artifacts directory: copy it separately when
transferring the repository. Previous feature caches are not needed.

## 1. Checks

```bash
continuation_root='experiments/issue-12-hidden-state-probes/issue-12-continuation-controls'
continuation_db='experiments/issue-12-hidden-state-probes/artifacts/db'

.venv/bin/python "$continuation_root/probe_controls.py" \
  --dataset-dir "$continuation_db" --audit-only

.venv/bin/python -m pytest -q "$continuation_root/test_continuation_controls.py"

.venv/bin/python -c 'import torch; print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

The audit should report 408/72/96 rows. Tests use a small randomly initialized
Qwen; they validate the implementation, not research performance.

The extractor loads the pinned model only from the local HF cache. If it is not
already cached, download it separately on the GPU machine:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-4B',
    revision='1cfa9a7208912126459214e8b04321603b3df60c',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
)
PY
```

## 2. Extract features on GPU

```bash
mkdir -p "$continuation_root/artifacts/logs"
set -o pipefail
.venv/bin/python -u "$continuation_root/extract_controls.py" \
  --dataset-dir "$continuation_db" \
  --output-dir "$continuation_root/artifacts/features-qwen3-4b" \
  --instruction none --fact-order shuffled \
  --layers 8,16,28,36 --device cuda --dtype bfloat16 \
  --batch-size 4 --max-length 512 \
  2>&1 | tee "$continuation_root/artifacts/logs/extract.log"
```

On OOM, reduce batch size to 1 and use a **new** output directory. Existing
directories are deliberately not overwritten. Automatic resume is not supported.
Truncation is forbidden. A continuation-length error requires checking the
tokenizer/protocol before running, rather than disabling the check.

Successful completion creates `features-qwen3-4b/manifest.json`, recording exact
continuations, the length audit, dataset hashes, feature/prompt hashes, model
revision, and library versions. An incomplete cache without this manifest cannot
be used to train probes.

## 3. Train probes and compute comparisons on CPU

```bash
.venv/bin/python -u "$continuation_root/probe_controls.py" \
  --dataset-dir "$continuation_db" \
  --feature-dir "$continuation_root/artifacts/features-qwen3-4b" \
  --output-dir "$continuation_root/artifacts/probes-qwen3-4b" \
  --instruction none --fact-order shuffled \
  --poolings last_token,mean_pool --l2-values 0.0001,0.001 \
  --max-iter 100 --device cpu --threads 1 \
  --seed 12012 --bootstrap-samples 2000 \
  2>&1 | tee "$continuation_root/artifacts/logs/probes.log"
```

This step does not need a GPU. All 70 probes are selected on train/validation
before test features are opened. Do not tune parameters on final test results.

## Outputs to inspect and share

In `artifacts/probes-qwen3-4b/`:

- `SUMMARY.md`: generated metric tables and paired differences;
- `report.json`: full report, configurations, per-target metrics, and intervals;
- `test_predictions.jsonl`: 960 rows (10 conditions × 96 situations);
- `selection.json`: validation choices recorded before opening test;
- 140 `.pt` files: 70 real-label and 70 shuffled-label classifiers.

Read `questions_vs_neutrals` and `general_vs_neutrals` first, then their contrasts
with each neutral control separately. Check whether neutral conditions improve
over `no_continuation`. An interval containing zero does not establish equality.

For initial review, share `SUMMARY.md`, `report.json`, `test_predictions.jsonl`,
and the feature `manifest.json`. Preserve the complete cache, checkpoints, and
logs for reproducibility. The recorded run is discussed in [RESULTS.md](RESULTS.md).
