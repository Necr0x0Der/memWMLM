# Research directions

This is a map of the **big directions**, not a list of experiments or results.
Individual experiments live inside these directions and should still have their
own issues, controls, and artifacts. Update this file when results change the
picture.

## Starting point: preliminary memory sanity check (not a direction)

The current `memWMLM` work is best read as a preliminary investigation, not as a
finished big direction.

It tested a small, resettable memory around a frozen LLM:

```text
source → writer → memory → reader → application
```

What it showed:

- source can be moved out of the application context, written into memory, and
  read back by the reader;
- with a frozen LLM, the reader must actually feed information into the model's
  input/processing path;
- in that setting, memory should be positionally ordered;
- the resulting memory is not obviously very different from a KV cache in terms
  of what it contains;
- nevertheless, the model can learn to solve specific tasks better, potentially
  better than LoRA, which is not very surprising given the larger capacity;
- when memory capacity is sufficient, the model does not try to memorize the
  training-set content in its own weights; it learns task-specific use of
  memory/context.

This is a sanity check. It establishes that the source → memory → application
channel can work and that positional memory is important. It does **not**
establish general semantic memory, durable memory, or a big architectural claim.

It leads to a fork between two directions below.

## Direction A — Modular skill extension: smaller LLM + learned modules

**Core question:** can a smaller LLM acquire skills at the level of a larger LLM
through modular extensions rather than through a uniformly larger parametric
body?

Possible modules:

- a pool of sidecar memory cells;
- a pool of sidecar experts;
- hierarchical or segmented memory;
- adapters/LoRA or other interface modules;
- combinations of memory and task-specific modules.

The current memory sidecar is an early instance of this direction. The open
question is not only "memory vs LoRA" but **which modular extension gives which
skill**, and whether the extension is reusable across tasks or merely adapts the
model to a particular task.

This direction is about learning/adaptation with gradients or trained side
parameters. It is distinct from Direction B's content storage without gradient
descent.

Open issues:

- what should be a module: memory, expert, adapter, routing/controller?
- how to match parameter and exposure budgets when comparing modules to a larger
  model or to LoRA;
- how to separate task adaptation from reusable skill acquisition;
- whether a module can generalize to novel instances rather than memorize the
  training set.

## Direction B — Internalized knowledge memory: writable memory without gradient descent on content

**Core question:** can knowledge that is absent from the smaller model's weights
but present in the larger model's weights be made **internal** to the model
through writable memory — more internal than RAG — without gradient descent on
the stored content, especially when the content cannot fit in context?

"External" here is only relative to the frozen LLM's weights: the knowledge is
outside those weights, but the target is not an external database or retrieval
tool. It should become internal runtime memory. The goal is therefore
**internalization**, not RAG-like external retrieval.

This direction separates:

- **content storage** — writing facts/episodes into memory;
- **interface/retrieval** — how a writer/reader queries that memory;
- **task adaptation** — whether the model learns to use the stored content.

The content itself is not learned by optimizer steps; it is written once or
updated by explicit operations. The **writing/reading procedure can still be
trainable**: writer/reader projections, routing, and task adaptation may require
gradient training. The DeltaNet variant is one attempt to minimize interface
training, but it does not completely avoid it; the frozen Qwen heads may need
adaptation, and retrieval still needs a learned or calibrated interface.

Initial experiments inside this direction:

- model-size / factual-knowledge comparisons (issue #5): whether 1.7B vs 4B
  gains are concentrated in factual recall rather than uniform reasoning;
- delta-rule / associative memory on TriviaQA (issue #6): write-once KV-like
  associative memory with a minimally trained interface, compared with RAG and
  full-context;
- RAG/full-context baselines with matched corpus, top-k, and prompt.

Important caveats:

- delta-rule state `S` is an associative KV memory, not a structured world state;
- capacity, decay, key collisions, positional binding, and interference;
- contamination and novel-instance leakage;
- matching RAG/full-context baselines and exposure budgets.

This direction is not "train a memory module" in the sense of learning the facts.
It is "write knowledge into internal runtime memory and retrieve it". The
knowledge is not injected by gradient descent, but the interface that writes and
reads it may be trained.

## Direction C — Content representations and computable world models

**Core question:** are LLM hidden states adequate as content/world state, or are
they mainly predictive features that are not suitable for semantic memory?

The critique:

- LLMs do not build descriptions of their input/context;
- they build features/embeddings useful for predicting subsequent tokens;
- these features are discriminative for next-token selection, not necessarily
  representative of the previous context;
- therefore they may be poor as a stable world state and poor as the content of
  semantic memory;
- a real system may need to change the training principle: learn **content
  representations**, not only predictive embeddings. Predictive embeddings do not
  have to be abandoned, but they should not be the only objective.

Related observations:

- `agimaze-predict` added a large state with explicit computational structure and
  improved spatial tasks. That is a hand-designed, fixed 2D workspace, not a
  learned content representation.
- DeltaNet/delta-rule state `S ← αS + βk⊗v` is closer to a working-memory content
  state than an embedding-per-token representation, but it is still a narrow,
  compressed KV/associative memory — not a world or problem state.
- JEPA predicts in latent space instead of reconstructing the input. A proposed
  extension is a **gated predictive autoencoder**: keep reconstruction loss but
  gate it, so the model learns content state rather than only predictive features.
  This idea is not yet documented or tested in the repository.

Direction C is connected to A and B:

- A needs a representation that modules can use;
- B needs content representations that can be stored and queried;
- neither is solved by adding capacity alone.

What this direction needs:

- a diagnostic probe: can frozen predictive hidden states be decoded into a
  known world state, and is that information causally usable?
- a minimal intervention: train a small model with an auxiliary content-state
  objective (explicit state reconstruction, structured entity slots, or a gated
  predictive autoencoder) and compare with next-token-only training;
- controls: shuffled/permuted state, null state, counterfactual state, novel
  instances, matched parameter/exposure budgets;
- evaluation of representation quality and downstream use, not only
  reconstruction accuracy.

## Cross-cutting principles

- Keep storage, retrieval, and task adaptation separate.
- Do not claim general semantic memory from a task-specific gain.
- Use `null_memory`, `shuffled_memory`, novel-instance splits, counterfactual
  source/state swaps, and matched exposure budgets where applicable.
- Record prompt and dataset provenance; do not silently relabel historical
  results.
- Frozen backbone is an experimental constraint, not a final architecture.
- Decodability is not the same as causal use.

## Relationship between directions

The preliminary memory work suggests a fork:

1. **A — modular skill extension**: a smaller model learns skills through
   modules/sidecars.
2. **B — internalized knowledge memory**: facts/knowledge are written into
   runtime memory without gradient descent on the content.

Direction C is not a third unrelated branch; it is the representation question
that underlies both. If predictive hidden states are poor content, then both
modular memory and internalized factual memory may be limited by representation,
not only by capacity or retrieval.

The current repository documents the preliminary memory work and fragments of
A, B, and C. It does not yet establish a complete big direction.
