# Running source-free memory on Qwen3-4B

Run all commands from the `memWMLM` repository root in Bash. Protocol and limits:
[README.md](README.md). Status of the previous attempt: [RESULTS.md](RESULTS.md).
This stage trains the adapter and inspects test answers; it does not swap memory.

## 1. Reuse existing features

**No extraction rerun is needed** if the previous Qwen3-4B experiment's features
were preserved. Soft-prefix training uses the same situation-only vectors as
the probes; no feature-format change is required.

Previous runbooks list two compatible sources:

- `experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b`
- `experiments/issue-12-hidden-state-probes/probe_training/artifacts/token-features-qwen3-4b`

The second cache also works: `--save-token-states` adds individual token states
but retains all three pooled representations. Individual tokens are not needed
here; large token-state tensors may slow shard loading.

Set the **actual feature path on the machine holding the cache**:

```bash
source_free_features='experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b'
source_free_dataset='experiments/issue-12-hidden-state-probes/artifacts/db'
source_free_output='experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/pilot-none-shuffled'
```

The manifest and every shard it lists for `train`, `validation`, and `test` are
required. A manifest alone, a report, or a trained probe checkpoint cannot
replace the feature cache. Qwen3-1.7B features are incompatible with this run.

Check the manifest:

```bash
.venv/bin/python - "$source_free_features" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / 'manifest.json').read_text())
assert m['model']['name'] == 'Qwen/Qwen3-4B'
assert m['model']['requested_revision'] == '1cfa9a7208912126459214e8b04321603b3df60c'
assert 28 in m['features']['layer_indices']
assert not m['tokenization']['chat_template_applied']
assert m.get('max_examples_per_split') is None
for split in ('train', 'validation', 'test'):
    shards = m['splits'][split]['shards']
    assert shards, split
    for shard in shards:
        assert (p / split / shard['path']).is_file(), shard['path']
    print(split, 'shards:', len(shards))
print('model:', m['model']['name'])
print('layers:', m['features']['layer_indices'])
print('pooling:', m['features']['pooling'])
PY
```

This is a quick file-presence check. During loading, `run.py` checks shard hashes,
dataset compatibility, and IDs. The pilot requires `mean_pool`, `last_token`,
and `sentence_end_mean`; additional cached layers are allowed.

## 2. Check GPU, model, and implementation

```bash
nvidia-smi
.venv/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
.venv/bin/python -m pytest -q experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/test_source_free.py
```

The answering pass still requires **Qwen3-4B weights**: cached features replace
only the repeated encoding of situations, not the answering model itself.
The script loads the pinned revision from the local Hugging Face cache and does
not download weights automatically. Reuse the HF cache from previous runs.
For an explicit download on a new machine with internet access:

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

VRAM is also needed for gradients with respect to the input: Qwen's weights
are frozen, but backward passes through it. Minimum VRAM for this run has not
yet been measured.

## 3. Only if the feature cache is missing: extract again

**Skip this step** if a compatible cache is available. Do not overwrite old
features; the extractor output directory must be new.

```bash
source_free_features='experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b'
.venv/bin/python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir "$source_free_dataset" \
  --output-dir "$source_free_features" \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 28 --device cuda --model-dtype bfloat16 \
  --output-dtype float16 --batch-size 4
```

The extractor saves all three poolings; `--save-token-states` is unnecessary.
Do not use `--max-examples-per-split`: it creates an incomplete smoke-test cache
that the main experiment deliberately rejects.

## 4. Train and generate test answers

Fixed pilot: `none/shuffled`, layer 28, three poolings, four memory slots,
five epochs, seed 12012. Select pooling and epoch by validation NLL. After
selection, generate test answers with memory, then without it. Test features
do not participate in adapter selection.

```bash
mkdir -p experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/logs
set -o pipefail
.venv/bin/python -u experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/run.py \
  --dataset-dir "$source_free_dataset" \
  --feature-dir "$source_free_features" \
  --output-dir "$source_free_output" \
  --instruction none --fact-order shuffled \
  --poolings mean_pool,last_token,sentence_end_mean --layers 28 \
  --slots 4 --epochs 5 --batch-size 2 --lr 1e-4 --seed 12012 \
  --device cuda --dtype bfloat16 --max-new-tokens 64 \
  2>&1 | tee experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/logs/pilot-none-shuffled.log
```

`source_free_output` must not exist before the run; its parent may exist.
Choose new output/log names for another attempt. The script reports
train/validation NLL after each epoch and progress every 50 test answers.
Expect 15 epoch records (3 poolings × 5), 672 answers with memory, and 672
without memory. The final `report.json` appears after both modes finish.

## 5. Inspect metrics and actual answers

```bash
.venv/bin/python - "$source_free_output" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads((p / 'report.json').read_text())
print('Selected:', r['selected'])
for mode, m in r['test'].items():
    print(mode,
          'macro:', round(100 * m['mean_macro_accuracy'], 2),
          'joint:', round(100 * m['joint_exact_targets_accuracy'], 2),
          'invalid:', m['invalid_answers'], '/', m['answers'])
for mode in ('memory', 'question_only'):
    print('\nExamples:', mode)
    with (p / f'test-{mode}.jsonl').open() as f:
        for _, line in zip(range(5), f):
            row = json.loads(line)
            print(row['id'], row['question'])
            print('Target:', row['target'], 'Prediction:', row['prediction'])
            print('Raw:', repr(row['response']))
PY
```

`macro` is mean balanced accuracy across seven questions; `joint` requires
all seven answers to be correct for one situation. Invalid answers count as
errors. Evaluation covers 96 situations but only four independent abstract layouts.

Preserve together: `protocol.json`, `history.json`, `selected.pt`, both answer
JSONL files, `report.json`, the log, and feature manifest/shards. Inspect invalids
and raw answers first. The difference from question-only is a preliminary signal
of interface usability; without swaps and an oracle, do not make a causal claim.
Earlier full-context QA used a different generation budget, so that comparison
is not matched.

## 6. If the run stops

- **CUDA unavailable:** run on a GPU machine; CPU tests do not replace 4B training.
- **Model not found locally:** check HF cache and pinned revision; perform the
  explicit download in step 2 if needed.
- **Feature/dataset mismatch or missing shard:** transfer the complete cache and
  its original dataset. Do not edit manifest hashes to bypass validation.
- **Output directory exists:** retain previous artifacts and choose a new name.
- **CUDA out of memory:** reduce batch size to 1 and start a separate run.
  Record this configuration change; it changes the number of optimizer steps.
- **No report.json:** the run is incomplete. History, the best checkpoint, and
  partial JSONL files may remain; they are not a complete test result.
- **Many invalid/truncated answers:** preserve the current evaluation. Treat
  a larger generation budget as a separate run, not a silent result replacement.

Automatic resume and a separate eval-only CLI are not implemented. Memory swaps,
an oracle, new instructions, and a larger layer grid are subsequent experiments,
not part of this pilot.
