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
