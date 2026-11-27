# Source-free soft-prefix memory: first test without the source description

Follow-up to [issue #12](https://github.com/Necr0x0Der/memWMLM/issues/12)
and [PR #14](https://github.com/Necr0x0Der/memWMLM/pull/14).
This is a separate protocol; previous experiments and artifacts remain unchanged.

Step-by-step instructions using existing features: [RUNBOOK.md](RUNBOOK.md).

## Research question

Can an affine adapter turn one saved situation vector into four virtual input
tokens that let **frozen Qwen3-4B answer questions about held-out test situations
without their source descriptions**?

```
situation without question → frozen Qwen → h at layer L with selected pooling
                                          ↓
                                 train-standardization
                                          ↓
                              trainable Linear(H, 4*D)
                                          ↓
                          [4 soft tokens] [chat question]
                                          ↓
                                 frozen Qwen → answer
```

Neither the source text nor its KV cache is passed to the answering pass.
Virtual tokens are supplied through `inputs_embeds`, before the complete chat
prompt. The adapter does not receive the question: all seven questions about
one world use the same memory. Training uses teacher forcing on
`FINAL: <answer>` and EOS, with loss on answer tokens only. Gradients flow
through Qwen to the adapter; Qwen's weights remain frozen. Features and their
normalization statistics receive no gradients; statistics use only training
rows from the selected condition. Adapter initialization uses the standard
deviation of Qwen's first 4,096 input embeddings.

This tests **representation usability through a learned interface**, not whether
the original Qwen used a prepared world state. The adapter is affine but has
many parameters: with H=D=2560 and K=4, it has 26,224,640 parameters. It may learn
part of the task. Four slots add no information to the original h; they learn
how to present that information.

## Pilot fixed before inspecting new test answers

- Model: Qwen/Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Existing dataset at `experiments/issue-12-hidden-state-probes/artifacts/db`,
  with unchanged splits.
- One condition, `none/shuffled`: 408 train, 72 validation, 96 test situations;
  seven questions per situation (2,856/504/672 answers).
- Layer **28** is fixed for the pilot. Pooling candidates: `mean_pool`,
  `last_token`, `sentence_end_mean`. The last averages the ends of six facts.
- One separate adapter per pooling; K=4, 5 epochs, AdamW lr=1e-4,
  weight decay=0, batch size=2, gradient clipping=1, seed=12012.
- Select pooling and epoch by **minimum validation answer-token NLL**.
  Ties favor the first candidate/earlier epoch. Do not select on test or
  retrain on train+validation.
- After selection, run greedy generation on all 96 test worlds and seven
  questions, with thinking disabled, max_new_tokens=64, and no constrained decoding.
- Primary metric: mean per-question balanced accuracy. Additional metrics:
  all seven targets correct jointly, per-question accuracy, invalid answers,
  and complete raw responses. This is not full world-state reconstruction.
- Control: the same frozen model, question, and prompt, but **without memory**.
  This control differs in context length and is not a matched null prefix.
- Initial positive signal: memory outperforms question-only and has a positive
  joint-target accuracy difference. This is a descriptive pilot result, not
  statistical proof of causal state use.
- Memory swaps, null prefixes, and oracle controls are **deferred at the user's
  request**. Without them, do not claim that the model causally uses the state.
- The existing test set has already been inspected in other experiments.
  There are only four independent test groups. Seed variation and confidence
  intervals are not estimated here.

The script supports multiple layers, such as `--layers 8,16,28,36`, but that is
a separate grid expansion and must be declared **before inspecting test results**.
Use a new output directory when changing conditions or budgets; do not overwrite
the pilot. Comparison with the earlier 95% full-context QA is not matched: that
run had a 2,048-token budget. Here, training targets short answers with a 64-token
limit. Many truncated/invalid answers limit the result to this protocol; do not
silently change the parser or generation budget after test evaluation.

## Running

Run from the repository root. Requires CUDA, locally cached pinned Qwen3-4B
weights, and enough VRAM for backward through the frozen backbone. Freezing
weights does not eliminate activation storage. The script does not download
weights automatically.

### 1. Features

Reuse a compatible complete Qwen3-4B feature cache with all three poolings and
layer 28 if available. Old probe reports/checkpoints do not replace features.
Otherwise, run the existing extractor, which saves all three poolings:

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 28 --device cuda --model-dtype bfloat16 \
  --output-dtype float16 --batch-size 4
```

### 2. Training, validation selection, and test answers

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/run.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/features-qwen3-4b \
  --output-dir experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/artifacts/pilot-none-shuffled \
  --instruction none --fact-order shuffled \
  --poolings mean_pool,last_token,sentence_end_mean --layers 28 \
  --slots 4 --epochs 5 --batch-size 2 --lr 1e-4 --seed 12012 \
  --device cuda --dtype bfloat16 --max-new-tokens 64
```

Every new run requires an output directory that does not already exist. The
script checks dataset and feature manifest/shard hashes, model/revision
compatibility, matching IDs, and disjoint abstract-layout groups. Test features
are loaded only after final selection; before that, test metadata is read only
for split auditing.

Outputs:

- `protocol.json`: configuration and provenance before training/evaluation;
- `history.json`: train/validation NLL for every candidate and epoch;
- `selected.pt`: best weights, train normalization, configuration, parameter count;
- `test-memory.jsonl`: generated answers with memory;
- `test-question_only.jsonl`: generated answers without memory;
- `report.json`: selected candidate, metrics for both conditions, invalids, hashes.

Test prompts contain no ground-truth answers. The correct answer is stored only
for scoring. The existing `FINAL:` parser is unchanged; invalid answers count
as errors. Generating one answer at a time simplifies checking that no cache
is transferred, but is not optimized for speed. Automatic resume is not
implemented; interruptions may leave history, the best checkpoint, and answers
already written.

### Implementation checks

```bash
.venv/bin/python -m pytest -q experiments/issue-12-hidden-state-probes/issue-12-source-free-memory/test_source_free.py
```

Tests use a small randomly initialized Qwen to check adapter gradients, frozen
backbone weights, loss/padding masks, absence of source text/targets in question
prompts, agreement with manual greedy generation, and the full
train→select→test pipeline. These are not Qwen3-4B performance measurements.

## Status and handoff

See [RESULTS.md](RESULTS.md) for the historical attempt and its limitations.
Run the pilot on a GPU machine with the pinned model and compatible features,
following this protocol.
