# memWMLM

Experimental infrastructure for **writable memory around a frozen language model**.

## Two-prompt exploratory inference

From this repository on a machine with the project dependencies and a trained **pooled-memory** checkpoint:

```bash
python3 scripts/generate_with_memory.py \
  --checkpoint runs/babi-memory/your-checkpoint.pt \
  --source 'Mary moved to the garden.' \
  --application 'Where is Mary? Answer:' \
  --max-new-tokens 64
```

`--source-file` and `--application-file` accept UTF-8 files instead of inline text. Each text is separately wrapped as a Qwen user turn with thinking disabled: the first is encoded once by the trained writer, then *only* the second prompt is placed in Qwen's generation context while the reader accesses the written memory. Source text is not replayed in the application context. The script prints the generated continuation, not the application prompt; use a checkpoint from `scripts/train_babi_memory.py` and only load trusted checkpoint files (PyTorch pickle). The bAbI-trained sidecar has not been trained for general conversation; outputs for arbitrary requests may be short, irrelevant, or otherwise unreliable. Use this as an exploratory probe rather than a measure of general-language quality.

The project starts deliberately below the long-term semantic-memory hypothesis. Its first target is a controlled reset-pair sanity check:

```text
source episode → writer → resettable memory cells → reader cross-attention → application episode
```

The axial language model remains frozen. The source text is unavailable in the application episode. Runtime memory is reset for each pair, so v0 does **not** test retention, organization, consolidation, revision, or multi-episode learning.

## QA1 20k training extension (same task, original test unchanged)

`datasets/qa1-extended-20k/qa1_merged_20k_train.txt` contains the original
10,000 training questions followed by 10,000 newly generated QA1 questions.
The extra 10,000 are also stored separately in the same directory. Use the
**unchanged** published test file under `artifacts/babi/tasks_1-20_v1-2/en-10k/`.
Both files use the existing bAbI parser. The generated half has 2,000 new
15-line episodes, each with 10 movements and five questions; actors, locations,
movement wording, question cadence and support recency were checked against the
published Task 1 sample. This is an *approximation* of its distribution, not
a reproduction of the unpublished v1.2 generator. In particular, do not
attribute a test-accuracy change solely to data volume without checking for
synthetic→published distribution shift. Details and SHA-256 hashes are in
`datasets/qa1-extended-20k/manifest.json`.

For the existing pooled-memory experiment, start from **your saved successful
early-only 10k config** (the exact 72.3% settings are not in this repository),
then override only the input and checkpoint destination:

```bash
python3 scripts/train_babi_memory.py \
  --config path/to/your-early-four-config.yaml \
  --train-file datasets/qa1-extended-20k/qa1_merged_20k_train.txt \
  --output runs/babi-memory/early-20k.pt
```

Verify that config still selects `[0,1,2,3]` and the same width, LoRA ranks,
batch and optimization settings as the 10k run. Evaluate with the **original**
test file (`--split test` for the primary comparison). For a fair learning
curve, record the number of presented examples (`steps × batch_size`) as well
as optimizer steps; random sampling with replacement does not correspond to
exact epochs. The evaluation vocabulary still comes from the original train
split via `babi_memory_eval.yaml`.

To reproduce the extension locally from downloaded bAbI v1.2 source files:

```bash
python3 scripts/extend_babi_qa1.py \
  --train-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt \
  --test-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --output-dir artifacts/babi/rebuilt-qa1-extension \
  --additional-examples 10000 --seed 20261020
```

The generator keeps the source question-blind and rejects generated examples
already present in the original train/test, as well as generated stories
matching the original test. The published train/test **already** share 16
exact `(story, question, answer)` records; the extension does not remove or
add to that pre-existing overlap.

### Nested 40k train extension

After the 20k train run, Alex reported ~77% accuracy on the unchanged
published test (versus up to 72.3% with 10k). To probe the same learning
curve, `datasets/qa1-extended-40k/qa1_merged_40k_train.txt` adds another
**20k** generated Task 1 questions. The existing 20k train is an **exact
byte-for-byte prefix**; the additional tranche is also in
`qa1_extra_20k_train.txt`. The original test remains untouched. The
`datasets/qa1-extended-40k/manifest.json` records the seed, input and output
hashes, counts and language/support-recency statistics; the same caveat about
synthetic versus published distribution applies.

Use the **same early-only configuration** as the 20k experiment and change
only `--train-file` and `--output` (do not overwrite its checkpoint):

```bash
python3 scripts/train_babi_memory.py \
  --config path/to/your-early-four-config.yaml \
  --train-file datasets/qa1-extended-40k/qa1_merged_40k_train.txt \
  --output runs/babi-memory/early-40k.pt
```

The prior 20k run and 40k run should be evaluated on the same original test
file, with optimizer steps and presented-example counts logged separately.
To regenerate the exact 40k file from the *original* train/test, invoke
`scripts/extend_babi_qa1.py` as above with `--additional-examples 30000`,
the same `--seed 20261020`, and a fresh `--output-dir`. The first 10k
generated questions remain identical to the 20k extension; the generator
rejects repeats against all accepted earlier examples.

## First milestone: bAbI Task 1

Dataset: bAbI Tasks v1.2, `en-10k/qa1_single-supporting-fact_{train,test}.txt`.

Before training memory, establish a frozen-Qwen full-context reference with a single fixed task prompt and no-thinking mode:

```text
[task instruction]
[source story]
[question]
Answer:
```

Score answers primarily by teacher-forced likelihood over the finite location vocabulary. This avoids conflating memory quality with free-form answer style.

Initial comparisons:

1. `no_source`: frozen Qwen receives only application prompt;
2. `full_context`: frozen Qwen receives source + application in ordinary context;
3. `memory`: source is committed into writable memory; application reads only memory;
4. `shuffled_memory`: application receives another example's committed memory.

Expected diagnostic ordering: `full_context ≫ no_source ≈ shuffled_memory`, with `memory` substantially above `no_source` and approaching `full_context`.

## Repository layout

```text
src/memwmlm/
  data/          Dataset download, parsers, source/application episode contract
  models/        Frozen axial-model wrappers and safe hidden-state hooks
  memory/        Writable cells, writers, readers, injection paths
  tasks/         Task prompts, candidate scoring, and task-specific evaluation
  training/      Reset-pair unroll, losses, optimizer/checkpoint loop
  evaluation/    Shared experiment variants and preservation controls
  runtime/       Stateful API and commit protocol (later; not needed by v0)
  utils/         Reproducibility, logging, serialization utilities

configs/
  data/          Dataset/split settings
  model/         Axial-model and hook placement settings
  memory/        Cell/writer/reader configurations
  experiment/    Fully composed, named experiments

docs/
  architecture.md  Stable conceptual and interface contracts
  research-plan.md Milestones, claims, and stopping criteria
  decisions.md     Short append-only architecture decisions

scripts/        Thin CLI entry points; no model logic
runs/           Local, ignored logs/checkpoints/metrics
artifacts/      Local, ignored datasets and downloaded model files
tests/          Unit/integration tests matching package boundaries
```

## Design rules

- Keep **data**, **memory architecture**, and **experiment protocol** separate. A task must not silently choose model hooks or memory shape.
- Use one canonical reset-pair batch contract for v0. Baselines and memory variants must consume the same source/application records.
- Keep the axial Qwen weights frozen in the initial study. Any task-format adapter or LoRA is a separately named experiment, not an implicit fallback.
- Memory values are runtime state produced by the writer; global writer/reader/cell parameters are ordinary trainable parameters. Do not mix these in checkpoint or optimizer code.
- Implement the simplest writable embedding-cell baseline before nonlinear experts. Later expert variants must retain the same external writer/reader contract whenever possible.
- Make ablations first-class configurations, not conditionals buried in training code.

Read [`docs/architecture.md`](docs/architecture.md) before implementing model code.

## Frozen-Qwen bAbI baseline

Install the project and its experiment dependencies in an environment with a
CUDA-compatible PyTorch build, then fetch the original bAbI archive:

```bash
python3 -m pip install -e '.[dev]'
python3 scripts/download_babi.py --output-dir artifacts/babi
```

Evaluate the two source-visibility controls. `--max-examples` is useful before
running the whole test split.

```bash
python3 scripts/evaluate_babi.py \
  --train-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt \
  --evaluation-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --variant full_context --max-examples 100

python3 scripts/evaluate_babi.py \
  --train-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt \
  --evaluation-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --variant application_only --max-examples 100
```

The default evaluator uses candidate likelihood: it renders Qwen's no-thinking
chat template, scores each bAbI location continuation after `Answer:`, and
reports exact candidate accuracy. To assess whether Qwen follows the one-word
output contract under ordinary greedy decoding, use:

```bash
python3 scripts/evaluate_babi.py \
  --train-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt \
  --evaluation-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --variant full_context --scoring free_generation --max-new-tokens 16
```

Free generation accepts only an exact candidate after case/outer-punctuation
normalization; it deliberately does not extract an answer from explanatory
prose. That makes formatting failures visible rather than silently turning
them into semantic successes.

## Frozen KV-cache boundary control (no training)

`evaluate_babi_kv_cache.py` compares the same full-context Qwen token sequence
processed in one forward pass with a two-pass continuation: prefill tokens
before `Question:`, retain Qwen's native layerwise KV cache, then process the
question, answer cue and candidate answer using that cache. The prefix is
chat-formatted and tokenized **once**, with a token-aligned split near the
question. Every candidate forks the original cache; no writer, reader, LoRA or
optimizer is involved. This is ordinary native attention across a processing
boundary, not independent semantic memory or a separate learned memory path.

```bash
python3 scripts/evaluate_babi_kv_cache.py \
  --train-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt \
  --evaluation-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --output runs/babi-kv-cache/control.json
```

By default, the evaluator processes the **entire evaluation split** and prints
aggregate full-context and cached candidate accuracy, prediction mismatches,
and maximum candidate-score difference. `--max-examples 20` is an optional quick
check; `--include-examples` adds individual predictions to JSON. For the first
three examples (`--verify-examples` controls the count) and up to three examples
with differing predictions (`--verify-mismatches`), it compares a single-example
full forward with cached computation at the source, application, and answer
logits. The JSON also reports batch-vs-single-full and single-full-vs-cache
candidate-score differences and prediction disagreements on verified examples,
plus a concise list of all disagreeing examples with their competing scores.
The full-context score uses the *same* batched scorer as `evaluate_babi.py
--variant full_context --scoring candidate_likelihood`. Minor floating-point
differences can arise from attention kernels; equality is numerical rather than
bitwise. A complete cache grows with source length and is not a bounded or
persistent memory mechanism.

## Learned episodic KV reader (experimental, GPU validation pending)

This experiment removes the Perceiver pooling writer. Frozen Qwen encodes a
source episode and saves **every source-token K/V at every layer**, independently
for each example. A learned cross-attention reader in the final four layers
reads the corresponding source K/V during a *separate application prompt*;
application Qwen does **not** receive the native source cache as its own
past_key_values. Only the readers and optional application q/o LoRA train.
This checks the learned interface, not bounded or durable semantic memory.

To test early access without changing the default late-four setup, pass
zero-based post-block `injection_indices` (e.g. `[0, 1, 2, 3]`); these select
both the reader hooks and the optional application LoRA projections. When
omitted, `injection_layers: 4` still selects the final four blocks, including
when loading earlier checkpoints. The indices are checked against the loaded
model's layer count and saved in new checkpoints. A checkpoint trained at
late layers cannot be loaded as an early-reader model: retrain with a separate
output path. The source encoder still saves K/V at **all** layers; only the
reader locations change.

```bash
python3 scripts/train_babi_episodic.py --config configs/experiment/babi_episodic_train.yaml
python3 scripts/evaluate_babi_episodic.py --config configs/experiment/babi_episodic_eval.yaml
python3 scripts/evaluate_babi_episodic.py --config configs/experiment/babi_episodic_eval.yaml --variant shuffled_memory
python3 scripts/evaluate_babi_episodic.py --config configs/experiment/babi_episodic_eval.yaml --variant null_memory
```

Early-four smoke run, with the evaluation checkpoint overridden (the eval
config defaults to the original late-four checkpoint):

```bash
python3 scripts/train_babi_episodic.py --config configs/experiment/babi_episodic_early_train.yaml
python3 scripts/evaluate_babi_episodic.py --config configs/experiment/babi_episodic_eval.yaml \
  --checkpoint runs/babi-episodic/reader-early.pt
```

The evaluation config selects both full train/test splits and candidate
likelihood. The source is reset for each pair, not carried across examples.
Training currently processes one unpadded pair per optimizer step (2,000
examples sampled with replacement in the example config). Results must be
interpreted alongside the frozen full-context, native-KV and pooled-cell
baselines; `memory` should beat `shuffled_memory`/`null_memory` on held-out
examples. A high score here alone cannot establish semantic memory.

A separate **task-adapted full-context** control trains rank-8 q/o LoRA in the
same final four layers on the same 2,000 sampled training pairs and answer+EOS
loss, with ordinary story+question context:

```bash
python3 scripts/train_babi_full_context_lora.py --config configs/experiment/babi_full_context_lora.yaml
python3 scripts/evaluate_babi_full_context_lora.py --config configs/experiment/babi_full_context_lora_eval.yaml
```

The control matches train split, sampled-example count, seed, LoRA rank/placement
and candidate-scoring contract, **not** parameter count or prompt topology: the
episodic reader adds parameters and uses two episodes. Compare train/test and
null/shuffled controls to distinguish task adaptation from source-specific
memory use. On this development host PyTorch/Transformers are absent; model
forwards and accuracy remain to be validated on the GPU computer. Begin with
small `--steps`/`--max-examples` smoke runs before the full runs.

## Reset-pair writable-memory sanity check

### Reader placement ablation

The original pooled-cell experiment supports `injection_indices` as a list of
zero-based Qwen decoder-block indices, for example `[0, 9, 18, 27]`. A reader
is added **after** each selected block; the next block can then process its
output. The selected indices also determine where optional source/application
LoRA adapters attach. With `injection_indices` omitted, the legacy
`injection_layers: 4` still means the last four blocks. Explicit indices take
precedence over `injection_layers` and are checked against the loaded model's
actual layer count at startup; old checkpoints lacking the field load as
before. New checkpoints store the exact indices for evaluation and debugging.

Use the same training budget/seed/settings as the late-four control and a
separate output path; an old late-layer checkpoint is **not** reusable with
new reader positions:

```bash
python3 scripts/train_babi_memory.py \
  --config configs/experiment/babi_memory_spaced.yaml

python3 scripts/evaluate_babi_memory.py \
  --config configs/experiment/babi_memory_eval.yaml \
  --checkpoint runs/babi-memory/reset-pair-spaced.pt \
  --scoring candidate_likelihood --split test
```

Compare late-four, spaced-four and (optionally) early-four placements at four
readers each. Report train/test and shuffled/null-memory controls. This tests
the location of the pooled-memory reader, not the uncompressed episodic KV
reader. Reader positions change model parameters, so train each from scratch.

The first learned-memory experiment keeps Qwen frozen and resets memory for
every bAbI pair. The source story is encoded, compressed by a trainable writer
into key/value cells, then removed. The application question accesses only
those cells through gated cross-attention injected into Qwen's final four
decoder layers.

Train the memory sidecar. All model and training defaults live in a flat YAML
config; explicit CLI flags override them:

```bash
python3 scripts/train_babi_memory.py \
  --config configs/experiment/babi_memory_train.yaml

# Example override without editing the recorded configuration:
python3 scripts/train_babi_memory.py \
  --config configs/experiment/babi_memory_train.yaml \
  --learning-rate 1e-4 --steps 10000 \
  --gate-bias-init -2 --max-grad-norm 1.0
```

Evaluate its intended path and two causal controls. By default the memory evaluator reports separate `train` and `test` objects; use `--split train` or `--split test` to run just one (and `--max-examples` limits each selected split):

```bash
python3 scripts/evaluate_babi_memory.py \
  --config configs/experiment/babi_memory_eval.yaml \
  --checkpoint runs/babi-memory/reset-pair.pt

# The committed state belongs to a different example; performance should fall.
python3 scripts/evaluate_babi_memory.py \
  --config configs/experiment/babi_memory_eval.yaml \
  --checkpoint runs/babi-memory/reset-pair.pt \
  --memory-variant shuffled_memory
```

Inspect several random source/application pairs before interpreting aggregate
accuracy. The debug runner prints the complete source story, application
question, free continuation, and all candidate likelihoods:

```bash
python3 scripts/debug_babi_memory.py \
  --config configs/experiment/babi_memory_train.yaml \
  --checkpoint runs/babi-memory/reset-pair.pt \
  --evaluation-file artifacts/babi/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt \
  --examples 5 --max-new-tokens 16
```

The train config supplies `model` and `train_file`; evaluation/debug still need
the checkpoint and test file because they vary per run. Training now includes
Qwen's EOS token in the supervised answer continuation, so the sidecar has an
explicit learned stop condition rather than only a location-word target. Reduce
`--max-new-tokens` (for example to `4`) to establish whether unexpectedly long
generations, rather than semantic errors, dominate free-generation evaluation.

### Application-only LoRA ablation

`configs/experiment/babi_memory_lora.yaml` enables rank-8 LoRA in `q_proj` and
`o_proj` of the same four upper Qwen layers used by memory injection. It is
**off** during the source pass and **on** only for the application pass; base
Qwen weights remain frozen. This tests whether Qwen can learn the memory
interface rather than requiring the sidecar alone to steer a fixed processor.

```bash
python3 scripts/train_babi_memory.py \
  --config configs/experiment/babi_memory_lora.yaml
```

### Source-side LoRA ablation

`configs/experiment/babi_memory_source_lora.yaml` instead enables a separate
rank-8 LoRA adapter only while encoding the source story. Unlike the original
writer path, this pass keeps an autograd graph from application loss through
writer into source LoRA (the base Qwen weights remain frozen). Application
LoRA is disabled in this config, isolating whether the source representations
can be made more useful *for writing* memory.

```bash
python3 scripts/train_babi_memory.py \
  --config configs/experiment/babi_memory_source_lora.yaml
```

The startup JSON reports independent `source` and `application` LoRA parameter
counts. A config with both ranks nonzero trains both scoped adapters.

For the no-memory capacity control, do not introduce a separate training mode:
train the same architecture with, for example, `--memory-width
32` (or evaluate `--memory-variant null_memory`). The essential comparison is
whether LoRA improves real `memory` substantially more than this reduced/null
memory condition.

This is a source-to-application channel check, not a claim of durable memory:
there is no state across examples, no automatic write policy, and no nonlinear
expert module yet.
