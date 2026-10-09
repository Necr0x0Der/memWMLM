# memWMLM

Experimental infrastructure for **writable memory around a frozen language model**.

The project starts deliberately below the long-term semantic-memory hypothesis. Its first target is a controlled reset-pair sanity check:

```text
source episode → writer → resettable memory cells → reader cross-attention → application episode
```

The axial language model remains frozen. The source text is unavailable in the application episode. Runtime memory is reset for each pair, so v0 does **not** test retention, organization, consolidation, revision, or multi-episode learning.

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

## Reset-pair writable-memory sanity check

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
  --learning-rate 1e-4 --steps 10000
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

This is a source-to-application channel check, not a claim of durable memory:
there is no state across examples, no automatic write policy, and no nonlinear
expert module yet.
