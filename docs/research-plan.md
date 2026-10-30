# Research plan

This repository supports a sequence of falsifiable claims. Do not skip a rung because a later design is more interesting.

## R0 — frozen axial-model interface

**Question:** Can frozen Qwen3 answer bAbI Task 1 under a stable prompt/output contract when it sees source and application in normal context?

- Backbone: `Qwen/Qwen3-1.7B`, frozen, no-thinking mode.
- Data: bAbI v1.2 `en-10k/qa1_single-supporting-fact`.
- Answer method: candidate log-likelihood over the observed location vocabulary, with a fixed task prefix.
- Output: recorded scores for `no_source` and `full_context`.

**Stop condition:** if the full-context reference is not meaningfully above no-source, fix prompt rendering or candidate scoring before adding memory. Do not train a memory path to compensate for a broken axial-task interface.

## R1 — resettable writable embedding-memory channel

**Question:** Can a trainable writer and reader communicate source information to a frozen Qwen application pass when source tokens are absent?

- Runtime memory is reset for every pair.
- Writer records source-derived keys/values into memory segments. (The current implementation is position-aligned: one cell per source position, so the cell count equals the source length rather than a fixed pool.)
- Application reads those segments via a gated cross-attention/residual pathway.
- Train only writer, cells, reader, and injection parameters.
- Compare all variants declared in `architecture.md`.

**Success signal:** `memory` materially exceeds `no_source` and `shuffled_memory`, while approaching the frozen `full_context` reference. Exact thresholds should be chosen after R0 establishes the difficulty scale.

**Non-claim:** R1 does not demonstrate long-term memory, knowledge organization, or nonlinear expertise. A solution that is latent source compression is valid for this rung.

## R2 — capacity and segment-addressing stress

**Question:** Does segmentation plus query-addressed memory retain usable information as source size/distractors exceed the application working-context budget?

Keep the same data semantics but vary source length, distractors, number of queried facts, number of cells, and tokens per segment. Compare a flat pooled-vector control against segmented memory. This tests the claim that attention over memory segments is useful before introducing nonlinear modules.

## R3 — multi-supporting-fact and compositional use

Move to bAbI Tasks 2 and 3. Source remains hidden in application. This tests whether the reader/writer interface can support multiple relevant traces, not merely a one-fact recall channel.

## R4 — persistent multi-commit memory

Replace reset pairs with short chains:

```text
source A → commit → source B → commit → delayed/shuffled application query
```

Begin with committed memory detached between steps, then assess limited unrolling. Add corrections/exceptions and delayed queries. Only here does retention, interference, update, and cross-episode composition become a legitimate claim.

## R5 — nonlinear memory modules

Hold the writer/reader external contract fixed and replace embedding values with mutable nonlinear expert state. The hypothesis is not that this must improve R1. It should be tested for compact compositional reuse, selective local updates, and reduced interference under R4-style mixed multi-commit streams.

## R6 — semantic transfer beyond bAbI

Use a generated or carefully isolated corpus of novel rules, counterfactual facts, versioned specifications, and application questions. It must contain assertions that conflict with the axial model's likely pretrained prior and must evaluate application, correction, uncertainty, and transfer—not only recall.

## Reproducibility requirements

Every reported run records:

- exact model revision, Transformers/PyTorch versions, and dtype;
- dataset source/checksum and parser version;
- prompt template and candidate vocabulary;
- composed config and random seeds;
- trainable parameter count and effective memory capacity;
- metrics for all declared controls; and
- a small qualitative error set, including shuffled-memory failures.
