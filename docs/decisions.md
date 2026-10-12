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

**Decision:** stop hyperparameter scaling of this pooled-cell branch. Preserve it as a source-specific-channel baseline. Next implement an episodic-attention control that transfers raw source cache/layerwise K/V in Qwen's native attention format across the source/application boundary, then test controlled compression of those K/V traces. This is an episodic-memory control, not the target semantic-memory architecture.
