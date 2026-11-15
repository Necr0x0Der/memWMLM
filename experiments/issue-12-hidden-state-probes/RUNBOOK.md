# Build the spatial-probe dataset, extract Qwen features, and fit linear probes

This runbook prepares the issue #12 dataset and extracts situation-only hidden
features from the frozen Qwen3-1.7B model, then fits the pre-specified shared
linear probes. Run every command from the repository root.

The feature run does not generate text, append questions, apply a chat template,
or update model weights. It saves last-token and mean-pooled representations for
transformer blocks 4, 8, 12, 16, 20, 24, and 28.

## 1. Get the experiment branch

On a new machine:

```bash
git clone https://github.com/Necr0x0Der/memWMLM.git
cd memWMLM
git fetch origin
git switch issue-12-frozen-llm-world-state
```

If the repository is already present, fetch and switch to the same branch. Do
not run the experiment from `main` or from a different commit. Record the exact
commit that will identify the run:

```bash
git status --short --branch
git rev-parse HEAD
```

The worktree should not contain local modifications to the generator,
extractor, dataset manifest, or model revision.

## 2. Prepare the GPU environment

The recommended machine has Linux, Git, `curl`, a working NVIDIA driver, a
BF16-capable NVIDIA GPU, and at least 8 GB VRAM. Qwen3-1.7B native BF16 weights
require roughly 3.4 GB before inference activations. Allow at least 8 GB free
disk space for the environment, model cache, generated dataset, and features.

Install the same pinned `uv` release used by the other Qwen experiment runbook:

```bash
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
```

Start a new shell so `uv` is on `PATH`, then create the repository-local
environment:

```bash
uv --version
uv python install 3.13
uv sync --extra dev --python 3.13
nvidia-smi
```

Verify PyTorch, Transformers, CUDA, BF16, and the issue #12 unit tests:

```bash
.venv/bin/python -c "import torch, transformers; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
.venv/bin/pytest -q \
  tests/test_spatial_probe_dataset.py \
  tests/test_hidden_feature_extraction.py \
  tests/test_linear_hidden_probes.py
```

Do not continue if CUDA or the tests fail.

## 3. Create the training/validation/test database

The generator is deterministic. Seed 12 and 24 episodes per abstract state are
the fixed settings for this protocol. Do not create the output directory before
running the command; the generator refuses to overwrite an existing dataset.

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/prepare_spatial_probe_dataset.py \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --seed 12 \
  --episodes-per-group 24
```

This produces:

```text
experiments/issue-12-hidden-state-probes/artifacts/db/manifest.json
experiments/issue-12-hidden-state-probes/artifacts/db/train.jsonl
experiments/issue-12-hidden-state-probes/artifacts/db/validation.jsonl
experiments/issue-12-hidden-state-probes/artifacts/db/test.jsonl
```

Verify the example counts, group counts, file hashes, and group separation:

```bash
.venv/bin/python -c "import json,pathlib; p=pathlib.Path('experiments/issue-12-hidden-state-probes/artifacts/db'); splits=('train','validation','test'); rows={s:[json.loads(x) for x in (p/f'{s}.jsonl').read_text().splitlines()] for s in splits}; groups={s:{r['group_id'] for r in rows[s]} for s in splits}; assert {s:len(rows[s]) for s in splits}=={'train':3264,'validation':576,'test':768}; assert {s:len(groups[s]) for s in splits}=={'train':17,'validation':3,'test':4}; assert all(not groups[a]&groups[b] for i,a in enumerate(splits) for b in splits[i+1:]); print({s:(len(rows[s]),len(groups[s])) for s in splits})"
echo "65f2123421fabe7bba8da39b0bf0c6e8b5930928fb0bda3293824601a371f448  experiments/issue-12-hidden-state-probes/artifacts/db/train.jsonl" | sha256sum -c -
echo "ffc60ba6068cba6648309322bda24adacffd62a6ddf213c4f15742321c4972f3  experiments/issue-12-hidden-state-probes/artifacts/db/validation.jsonl" | sha256sum -c -
echo "ae2a052f5d95493b3d719853d8f430a7b2f4f3f3180a8029281cfcd2b1c27e89  experiments/issue-12-hidden-state-probes/artifacts/db/test.jsonl" | sha256sum -c -
```

Expected output:

```text
{'train': (3264, 17), 'validation': (576, 3), 'test': (768, 4)}
```

The independent split unit is the abstract person-rank layout. Room-name
permutations, fact-order variants, and instruction variants of a layout must
never cross splits.

## 4. Download the pinned Qwen3-1.7B model

Download the exact model revision already used by the repository's Qwen
experiments:

```bash
.venv/bin/hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

The extractor uses `local_files_only=True` by default. It will fail instead of
silently downloading a different revision.

Verify the cached configuration before the full run:

```bash
.venv/bin/python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', local_files_only=True); assert c.num_hidden_layers==28 and c.hidden_size==2048; print(c.model_type, c.num_hidden_layers, c.hidden_size)"
```

Expected output begins with:

```text
qwen3 28 2048
```

## 5. Select a GPU and extract the hidden features

Inspect the available GPUs:

```bash
nvidia-smi
```

Replace the first `0` below with the physical number of a free GPU. Within the
process, the selected device is exposed as `cuda:0`.

### 5.1 Run a six-example model smoke test

This processes the first two records from each split. It verifies the real
model load and forward path but must not be used for probe results. Do not
create the smoke output directory beforehand.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b-smoke \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --layers 4,8,12,16,20,24,last \
  --device cuda \
  --model-dtype bfloat16 \
  --output-dtype float16 \
  --batch-size 2 \
  --max-length 256 \
  --max-examples-per-split 2
```

Verify that all parameters remained frozen and exactly six examples were
processed:

```bash
.venv/bin/python -c "import json,pathlib; m=json.loads(pathlib.Path('experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b-smoke/manifest.json').read_text()); assert m['model']['trainable_parameters']==0; assert m['max_examples_per_split']==2; assert {s:m['splits'][s]['examples'] for s in ('train','validation','test')}=={'train':2,'validation':2,'test':2}; print(m['model']); print('peak_vram_bytes',m['peak_vram_bytes'])"
```

### 5.2 Run the complete extraction

Do not create the full feature output directory beforehand.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --layers 4,8,12,16,20,24,last \
  --device cuda \
  --model-dtype bfloat16 \
  --output-dtype float16 \
  --batch-size 8 \
  --max-length 256
```

The extractor loads Qwen with the same `AutoModelForCausalLM.from_pretrained`
path used by the existing Qwen experiments, freezes every parameter, and calls
only the base `.model`. It does not execute the language-model head or produce
next-token logits.

The output contains a manifest and bounded-size PyTorch shards:

```text
experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b/manifest.json
experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b/train/shard-*.pt
experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b/validation/shard-*.pt
experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b/test/shard-*.pt
```

With float16 storage, the two pooling variants occupy approximately 252 MiB of
raw tensor data, plus serialization overhead.

## 6. Verify the feature artifacts

Check the frozen-model record, layer selection, split sizes, shard hashes, and
tensor shapes:

```bash
.venv/bin/python -c "import hashlib,json,pathlib,torch; p=pathlib.Path('experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b'); m=json.loads((p/'manifest.json').read_text()); assert m['model']['trainable_parameters']==0 and m['model']['eval_mode']; assert m['model']['requested_revision']=='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'; assert m['features']['layer_indices']==[4,8,12,16,20,24,28]; expected={'train':3264,'validation':576,'test':768}; assert all(m['splits'][s]['examples']==n for s,n in expected.items()); shards=[(s,x) for s in expected for x in m['splits'][s]['shards']]; assert all(hashlib.sha256((p/s/x['path']).read_bytes()).hexdigest()==x['sha256'] for s,x in shards); sample=torch.load(p/'train'/m['splits']['train']['shards'][0]['path'],weights_only=True); assert sample['last_token'].shape[1:]==(7,2048) and sample['mean_pool'].shape[1:]==(7,2048); assert sample['last_token'].dtype==torch.float16 and sample['ids']; print(m['model']); print({s:m['splits'][s]['examples'] for s in expected}); print(sample['last_token'].shape, sample['mean_pool'].shape)"
```

The first tensor dimension equals the number of examples in that shard; with
the default batch size the printed sample shape should normally be
`[8, 7, 2048]` for both pooling variants.

## 7. Preserve and transfer the run

Keep these artifacts together:

- `artifacts/db/manifest.json` and all three database JSONL files;
- the complete `features-qwen3-1.7b/` directory;
- the terminal log containing GPU model, runtime, and any peak-memory data;
- the repository commit SHA printed in step 1.

The extractor publishes the output directory only after all three splits
finish. It is not currently resumable shard-by-shard. If it is interrupted, run
the exact same command again after confirming that the final output directory
does not exist. Do not merge shards from commands with different commits,
model revisions, layer lists, precision, tokenization, or dataset manifests.

Do not inspect test metrics outside the command below. The training script
selects the shared pooling/layer/L2 configuration from validation and touches
test only after that choice is locked.

## 8. Select the shared probe bank and evaluate once

The probe script standardizes every feature dimension using train statistics,
fits one linear classifier per fixed question, and selects one common pooling,
layer, and L2 coefficient. Run it once for the result-bearing feature set:

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/train_linear_probes.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/artifacts/features-qwen3-1.7b \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/probes-qwen3-1.7b \
  --l2-values 0,1e-5,1e-4,1e-3 \
  --max-iter 100 \
  --device cpu \
  --shuffle-seed 12012 \
  --bootstrap-samples 2000
```

The command refuses to overwrite an existing output and verifies all dataset
and feature hashes. It writes:

```text
experiments/issue-12-hidden-state-probes/artifacts/probes-qwen3-1.7b/report.json
experiments/issue-12-hidden-state-probes/artifacts/probes-qwen3-1.7b/selected_probe.pt
```

Check the locked selection and headline metrics without modifying the report:

```bash
.venv/bin/python -c "import json,pathlib; p=pathlib.Path('experiments/issue-12-hidden-state-probes/artifacts/probes-qwen3-1.7b/report.json'); r=json.loads(p.read_text()); print('selected',r['selected']); print('test',r['evaluation']['test']['overall']); print('shuffle',r['shuffled_label_control']['test']['overall'])"
```

Interpret this stage as linear decodability only. The test report must include
the joint exact-state score and the instruction/order breakdowns, not only the
best individual pairwise probe. Source-free causal use and full-context Qwen
accuracy remain separate controls/stages.
