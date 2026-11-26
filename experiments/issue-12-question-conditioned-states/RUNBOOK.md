# Runbook: states after the situation and question, before generation

Run all commands from the repository root in Bash. Protocol: [README.md](README.md).
Requirements: the existing `.venv`, locally cached pinned Qwen3-4B weights, and
the original dataset. Feature extraction uses a GPU; probe training runs on CPU
and does not load Qwen.

## 1. Audit and tests

```bash
.venv/bin/python experiments/issue-12-question-conditioned-states/probe.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --audit-only

.venv/bin/python -m pytest -q \
  experiments/issue-12-question-conditioned-states/test_question_states.py

nvidia-smi
.venv/bin/python -c 'import torch; print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

The audit should report `train: 408, validation: 72, test: 96` for `none/shuffled`.
The tests use a small randomly initialized model and do not measure 4B performance.

## 2. Extract features for both conditions

**The previous pooled features are not a matched baseline for this comparison.**
They were extracted without a chat template or a question. The new states cannot
be reconstructed from the old mean or last-token vectors. This extractor creates
both the no-question baseline and question-conditioned features with the same
chat formatting. Keep the previous files. Cached pinned weights can be reused
without downloading the model again.

```bash
mkdir -p experiments/issue-12-question-conditioned-states/artifacts/logs
set -o pipefail
.venv/bin/python -u experiments/issue-12-question-conditioned-states/extract.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-question-conditioned-states/artifacts/features-qwen3-4b \
  --instruction none --fact-order shuffled \
  --layers 8,16,28,36 --device cuda --dtype bfloat16 \
  --batch-size 4 --max-length 512 \
  2>&1 | tee experiments/issue-12-question-conditioned-states/artifacts/logs/extract.log
```

The output directory must not already exist. The script loads
`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` with `local_files_only=True`.
If the weights are not cached, run this separately on a machine with internet access:

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

Cache layout:

```
features-qwen3-4b/
  protocol.json
  manifest.json                  # written only after extraction completes
  no_question/
    train/prompts.jsonl
    train/shard-00000.pt ...
    validation/ ...
    test/ ...
  mary_left_of_john/
    train/ ...
    validation/ ...
    test/ ...
  ... the other six questions
```

Each prompt ends with the assistant prefix, **before the first generated token**.
It contains no ground-truth answer, answer options, or generated reasoning.
`mean_pool` includes the situation, question when present, and chat tokens;
`last_token` is the final token of the assistant prefix. It is the last input
position, not the last word of the situation or question. Both conditions use
the same template ending. `sentence_end_mean` is not used in this protocol.
Pooled features are saved in FP16; the full per-token trace is not saved.

Inspect the exact rendered prompts:

```bash
head -n 1 experiments/issue-12-question-conditioned-states/artifacts/features-qwen3-4b/no_question/train/prompts.jsonl
head -n 1 experiments/issue-12-question-conditioned-states/artifacts/features-qwen3-4b/mary_left_of_john/train/prompts.jsonl
```

## 3. Train probes and evaluate on test

```bash
set -o pipefail
.venv/bin/python -u experiments/issue-12-question-conditioned-states/probe.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-question-conditioned-states/artifacts/features-qwen3-4b \
  --output-dir experiments/issue-12-question-conditioned-states/artifacts/probes-qwen3-4b \
  --instruction none --fact-order shuffled \
  --poolings last_token,mean_pool --l2-values 0.0001,0.001 \
  --max-iter 100 --device cpu --threads 1 \
  --seed 12012 --bootstrap-samples 2000 \
  2>&1 | tee experiments/issue-12-question-conditioned-states/artifacts/logs/probe.log
```

Expect eight `locked_context` messages. After selecting all 56 classifiers,
the script loads test features and computes metrics and paired differences.
Qwen does not generate answers at this stage: the probes predict the targets.

Outputs:

- `protocol.json`: settings recorded before selection and evaluation;
- `selection.json`: selected configurations and all validation candidates;
- `context--target.pt`: selected classifier and train-only normalization;
- `context--target--shuffled-label.pt`: negative-control classifier;
- `test_predictions.jsonl`: 768 rows, each containing seven predictions;
- `report.json`: context metrics, controls, paired differences, and provenance.

## 4. Inspect the comparison

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/issue-12-question-conditioned-states/artifacts/probes-qwen3-4b/report.json')
r = json.loads(p.read_text())
print('Baseline macro:', r['contexts']['no_question']['test']['mean_macro_accuracy'])
print('Differences below are in percentage points.')
for question, effects in r['effects'].items():
    print('\nQuestion:', question)
    for name in ('asked', 'inverse', 'other_excluding_inverse', 'joint_exact_targets'):
        if name in effects:
            e = effects[name]
            print(name, round(100*e['delta'], 2),
                  [round(100*x, 2) for x in e['paired_group_bootstrap_95ci']])
print('\nBalanced accuracy matrix (%):')
names = list(r['contexts']['no_question']['test']['per_probe'])
print('context', *names, sep='\t')
for context, result in r['contexts'].items():
    print(context, *[round(100*result['test']['per_probe'][n]['macro_accuracy'], 2)
                     for n in names], sep='\t')
PY
```

An improvement on the diagonal (`asked`) means the requested target is more
decodable after the question. Improvement on the inverse relation is not
independent evidence. Inspect `other_excluding_inverse` and shuffled-label
controls as well as the diagonal. There are only four independent test groups,
so statistical conclusions remain exploratory. Probe success does not establish
that Qwen itself uses the representation when answering.

## 5. Troubleshooting and reruns

- CUDA or weights unavailable: run extraction on a prepared GPU machine.
- Out of memory during extraction: reduce the batch size and use a new output
  directory and log. The protocol stays the same, although numerical results
  may differ slightly.
- Prompt exceeds max-length: increase the limit and rerun extraction in a new
  directory. Prompts are never silently truncated.
- Final manifest or report missing: the run has not completed. Do not treat
  partial output as a complete result.
- Hash, ID, or condition mismatch: verify the dataset and complete feature cache;
  do not edit hashes to bypass validation.
- Output directory already exists: preserve the old artifacts and choose a new name.

Resume is not implemented. Do not expand the layer/L2 grid after inspecting test
results and present it as the same confirmatory comparison. Preserve the feature
cache, checkpoints, manifests, logs, predictions, and report. Do not overwrite
historical probe or memory-adapter results.
