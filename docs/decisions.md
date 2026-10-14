# Architecture decisions

Concise record of decisions that govern experiments in this repository.

## 2026-10-09 — Axial backbone family

Start with dense Qwen3, not Qwen3.5 or later hybrid backbones.

- First end-to-end development backbone: `Qwen/Qwen3-1.7B`.
- First meaningful replication: `Qwen/Qwen3-4B`.
- Rationale: Qwen3 is a conventional dense causal decoder with an easily controlled KV cache and injection points. Qwen3.5's Gated DeltaNet/full-attention hybrid introduces unrelated questions about where and how to insert memory.
- Constraint: memory code consumes/returns hidden states through an axial-model interface; do not hard-code Qwen3 attention implementation throughout the project.

## 2026-10-09 — v0 memory protocol

Use a resettable two-episode meta-learning pair:

```text
source → commit → application
```

- Runtime memory resets before every pair.
- The source is processed by normal causal attention and cannot read memory.
- The application cannot see source tokens; it can only access the committed state through a memory reader.
- Only application loss is supervised.
- Axial model weights are frozen.

This is a writer/reader plumbing sanity check, not evidence of persistent semantic memory.

## 2026-10-09 — v0 memory representation

Begin with a pool of writable embedding-memory segments, not nonlinear experts.

- Writer emits source-derived keys and values.
- Reader uses additional learned query/key/value projections to attend to segments.
- Returned features enter selected upper axial layers through a zero-initialized gated residual path.
- Nonlinear mutable experts are a later substitute behind the same writer/read interface, to be tested on multi-commit interference/composition tasks.

## 2026-10-09 — first task and axial-model control

Use bAbI Tasks v1.2, English 10k Task 1:

```text
tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{train,test}.txt
```

First establish frozen Qwen performance with source plus application in ordinary context. Use one fixed bAbI instruction, no-thinking mode, and candidate likelihood over the finite location vocabulary. Do not initially add LoRA or a task-format adapter; that would be a separately declared experiment if the frozen prompt interface proves inadequate.

## 2026-10-12 — Result: pooled embedding-memory reset-pair baseline

The first implementation used a Perceiver-style writer and late residual reader:

```text
source final Qwen states
→ 32 learned write queries attend over all source positions
→ one 512D key/value cell per query
→ application-only reader cross-attends to cells
→ gated residual injection into the final 1–4 Qwen layers
```

- bAbI Task 1 full-context frozen Qwen: about 82% accuracy; application-only: 0%.
- Pooled memory plateau: about 60% training / 55% test accuracy; candidate likelihood slightly exceeded free generation after output formatting/EOS was fixed.
- `null_memory` and `shuffled_memory` were both about 17%, so the learned runtime state is source-specific and necessary, but this does not establish an adequate episodic/world/semantic representation.
- Increasing cell count, memory width, attention-head count, or injection layers did not materially change the plateau; one injection layer was close to four. Training loss remained noisy and did not converge to a near-zero train solution.
- Separate source-side and application-side rank-8 LoRA adapters made optimisation more stable and raised both train and test by only about 2.5 percentage points. This confirms a small interface-adaptation effect but does not solve the bottleneck.

Interpretation: the limitation is structural rather than simple capacity. The writer recomputes query-agnostic global summaries; it has no sequential allocation, fact/entity slots, overwrite operation, explicit positional/event trace, or direct cell supervision. The reader also enters late through a new residual channel rather than Qwen's native layerwise attention K/V path.

**Decision:** stop hyperparameter scaling of this pooled-cell branch. Preserve it as a source-specific-channel baseline. Next implement an episodic-attention control that transfers raw source cache/layerwise K/V in Qwen's native attention format across the source/application boundary. This is an episodic-memory control, not the target semantic-memory architecture. Subsequent memory representations remain open for discussion; compression is a possible experiment, not a settled objective.

## 2026-10-13 — First KV control: frozen native continuation

Implement the first, **non-training** control: tokenize the complete full-context prompt once, prefill the native Qwen KV cache through the story, then continue with the question/answer and score candidates using that cache. Compare logits and candidate predictions with a single-pass full-context computation on exactly the same tokens, including position handling. No trainable writer, reader or adapter is involved. This verifies the processing boundary, not a learned memory interface or semantic knowledge acquisition. The later learned-memory experiments also measure task learning; frozen full-context accuracy is not an upper bound on an adapted model.

Alex's full bAbI Task 1 test: frozen `full_context` candidate accuracy 81.9%, native-KV continuation 82.4% across 1,000 examples; 15 predictions differ. These are two candidate-likelihood evaluations, not free-generation versus candidate scoring. A checked disagreement already shows source logits differ between full and short-prefix passes, consistent with numerical/kernel differences, while exact equivalence remains unproven. Treat this as approximate performance parity, not a statistically meaningful KV improvement or a semantic-memory result.

## 2026-10-13 — Learned uncompressed episodic side-path (experimental design)

Next test a separate trainable reader over all source-token per-layer Qwen K/V without pooling or compression. The source is encoded with frozen Qwen and is not present in the application context; the application starts its own positional context and may access source only through gated cross-attention side paths in the last four layers. Train reader plus optional application-side rank-8 LoRA, while keeping Qwen base/source encoder frozen. Measure candidate accuracy on train/test and shuffled/null-memory controls. Alongside it, train a frozen-base full-context rank-8 LoRA task control on the same bAbI task, answer+EOS loss, train examples/optimizer steps and upper q/o projection placement. These are not identical parameter budgets: the side reader has additional parameters and sees a separate source prompt. A result above frozen 82% can reflect task adaptation, not memory quality alone. The exact future representation beyond this episodic upper-bound probe remains open.

## 2026-10-14 — Test early and spaced injection in pooled-cell memory

Alex reports that episodic KV-reader memory reached about 33% after 2,000
single-example steps, then about 55–56% with more intensive training; it is
**not** a 56% result after only 2,000 steps. The task-adapted full-context LoRA
reached about 98% train / 94% test. These are reported runs, not evidence that
the episodic model has converged. Both learned memory variants so far inject
only at late layers. A late-one versus late-four comparison does not test
whether early memory access lets the model combine the question and source.

Add explicit zero-based `injection_indices` to the original *pooled embedding
memory* experiment; retain `injection_layers` as the old last-N default. Keep
the number of reader positions equal for late-four versus spaced-four
(`0,9,18,27` when those layers exist) and possibly early-four; compare with
the same training budget and shuffled/null-memory controls. Hooks add memory
**after** the selected block. A later run will determine whether placement,
as opposed to writer/representation or training, explains the gap.
