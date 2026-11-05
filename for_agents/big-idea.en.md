# Project big idea

This is an informal description of the long-term hypothesis, adapted for agents.
It is not a proven claim and should be updated after experiments.

## 1. Initial hypothesis

LLMs should not be so gigantic. They mix:

- factual knowledge;
- general reasoning ability;
- cached solutions, as in AlphaGo where hundreds of millions of games are embedded;
- broad coverage of human knowledge.

Such LLMs can be effective as "reflexes" or crystallized intelligence on tasks they were trained on. But they:

- solve new tasks poorly;
- do not change without retraining;
- store weights for knowledge that is unnecessary for other domains;
- conflict with new knowledge when facts must be updated.

A human does not need such a volume of information for effective thinking. Therefore it should be possible to reduce an LLM by moving part of its knowledge into **rewritable memory** without degrading reasoning.

If factual knowledge can be moved into semantic memory, the model could:

- keep knowledge current without retraining;
- work over a longer horizon;
- reduce conflict between frozen weights and new knowledge;
- support life-long learning and personalization;
- potentially be smaller, or reason better at the same size.

## 2. Memory is not just a tape of embeddings

The problem is deeper than "context extension." Memory should be a system of representations tied to **computable world models**:

- graphs;
- spatial maps;
- event structures;
- rules;
- object and relation models.

LLMs do not form such computable representations internally; they emulate them through text descriptions. But text does not replace computation over structures — for example, spatial attention or tracing a path.

Therefore the final system should not be merely "frozen LLM + reader/writer to memory." Reasoning should be done in terms of computable world models.

## 3. Two directions

### Direction 1: computable representations / world models

`agimaze-predict` added a visual workspace and showed that an LLM can solve spatial tasks more effectively. This is not an arbitrary world model, but a fixed 2D representation. It can be extended:

- different structure types;
- training the model to execute code that creates data structures;
- persistent maps;
- cross-size comparison of a small model with workspace versus a large model.

This is important, but it does not directly test the hypothesis that LLMs should not be so large.

### Direction 2: small LLM + rewritable memory

`memWMLM` directly tests the hypothesis: a small LLM plus memory may be comparable to a larger LLM.

Its memory was initially primitive: pooled embedding cells, writer/reader, reset-pair. This does not prove the memory architecture, but it demonstrates the source → memory → application channel.

## 4. What experiments show as of 2026-11-05

### memWMLM

- Frozen Qwen3-1.7B on bAbI QA1 full-context ≈ 82%; application-only ≈ 0%.
- Pooled memory ≈ 60% train / 55% test; `null_memory` and `shuffled_memory` ≈ 17%.
- LoRA adapters gave only ~+2.5 pp and were effectively necessary.
- Early injection `[0,1,2,3]` was better than late injection: pooled early ≈ 72.3% test; late + LoRA ≈ 50–60% test.
- QA2: frozen full-context 49.6%; pooled early memory 99.4% train / 81.6% test.
- Six bAbI tasks: per-task checkpoints were better than mixed; mixed was lower on five of six tests.
- 40k QA1 train: 99.7% on the unchanged test, but this is a same-task synthetic extension, not general memory.
- Qwen3-4B on bAbI: not a general improvement; substantial improvement only on QA4, a small improvement on QA12, a small regression on QA11.

### Position-aligned memory and repro

- Pooled "bag" memory lost positional/event structure.
- Position-aligned writer: one cell per source position; cell count = `src_len`.
- Repro Stage A (WikiText-103): 120k train windows, 3,092 held-out.
- Memory width 2048: repro-loss stuck at ~0.2–0.3 — a capacity limit.
- Width 4096: loss fell nearly to zero; eval exact ~0.2, chain ~82.
- Width 2048: exact ~0.005, chain ~16.
- 4096 repro checkpoint → bAbI: qa1 0.984, qa4 1.0, qa12 0.964, qa11 0.994; qa2 0.824, qa16 0.894.

### Baseline Qwen on repro

- Qwen3-1.7B full-context, chain: exact 0.08, chain 21.
- Qwen3-4B: exact 0.06, chain 18.
- So 4B is not better on this task; base Qwen copies only a short prefix.
- Memory 2048 was around chain 16 / exact 0.005 — not above baseline.
- Memory 4096 was chain 82 / exact 0.2 — above baseline.

### Selective reproduction

- Memory 4096, trained at `whole_prob 0.3`, evaluated sentence-only at `whole_prob 0.0`:
  - exact 0.42, chain 17;
  - baseline Qwen 1.7B: exact 0.02, chain 2.
- Higher exact on short sentences is mostly a target-length effect.
- Selective reproduction was harder than whole-passage reproduction for both models.
- Selective training loss plateaued/noisy and did not fully converge.
- WikiText is noisy: headers, `@-@`, truncation, regex splitting.

## 5. Clarifications after experiments

### 5.1. Episodic memory ≠ semantic memory

Episodic memory can be raw or almost uncompressed. Compression is for cost, not necessarily for the ability to store.

Semantic memory is more like a **smart index to episodes**: entities, facts, events, versions, pointers, relations. The main problem is **retrieval**, not storage.

### 5.2. The current model is a memory-use competence

The memory sidecar + LoRA learns not only to store, but also:

- what to write;
- how to read;
- how to inject into layers;
- how to use source/memory in a task.

Therefore improvement over base Qwen is not a pure memory effect, but **memory + a learned competence for using it**.

### 5.3. Frozen backbone is an experimental constraint, not dogma

A fully frozen LLM is used to separate memory from fine-tuning. It is not required for a real system. If LoRA is needed to use memory, the claim should be "small model + memory + small adaptation," not "memory fully replaces training."

### 5.4. Tasks must actually require memory

bAbI QA1 is lookup-friendly and saturated. Selective repro tests positional addressing, but not multi-hop, temporal, update, or contradiction.

Tasks should require:

- retrieval without source in context;
- multi-hop;
- temporal order;
- aggregation;
- update/versioning;
- novel instances;
- counterfactual source swap.

## 6. Open questions and next steps

1. Selective repro — a diagnostic rung: retrieval vs generation.
2. Repro pause: either cleaner texts or move to semantic-memory/quiz.
3. Stage B: dense no-reasoning QA over a memorized corpus.
4. LwF: frozen backbone + memory filled by one reading of a dialect; novel-instance; memory vs LoRA/ICL/RAG.
5. Factual knowledge quiz: MMLU/TriviaQA/NQ/SimpleQuestions/Wikidata; a separate claim about large factual memory.
6. Persistence/update: multi-commit, overwrite, interference.
7. Architecture: hierarchical/segment memory, semantic index, query controller, non-frozen adapters.
8. Model-size ladder: 1.7B+memory vs 1.7B+LoRA vs 3B/7B full-context.

## 7. Main conclusion

The hypothesis "small LLM + rewritable memory ≈ large LLM" is **not proven yet**. What is shown:

- the source → memory → answer channel works;
- positional memory is important;
- capacity 2048 vs 4096 is a real limit;
- memory + competence can strongly improve base Qwen;
- bAbI shows memory use, but in a task-specific regime.

The next stage should test **retrieval, generalization, write-once, and update**, not only reproduction.
