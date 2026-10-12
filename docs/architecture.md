# Architecture contract

This document specifies interfaces, not a claim that the v0 memory is semantic or long-term.

## Scope of v0

The v0 experiment is a **reset-pair sanity check**:

```text
M0 (same initial state for every example)
source ──► frozen axial Qwen ──► writer ──► M1
application ──► frozen axial Qwen + reader(M1) ──► answer loss
```

- The source is available only through Qwen's ordinary within-episode attention.
- `M1` is reset before each source/application pair.
- The application cannot access source tokens.
- The axial Qwen weights are frozen.
- Writer, memory-cell projections, reader, and memory injection are trainable.
- Runtime state `M1` is produced in the forward pass; it is not an optimizer parameter.

Therefore v0 establishes only that the learned source-to-memory-to-application channel works. It does **not** establish durable retention, multi-episode organization, semantic consolidation, automatic writes, or a persistent agent.

## Canonical data contract

All v0 variants consume the same record:

```python
@dataclass
class ResetPair:
    example_id: str
    source_text: str
    application_prefix: str   # fixed task prefix + question + answer cue
    target: str               # canonical answer, e.g. "kitchen"
    candidates: tuple[str, ...]
    metadata: dict[str, object]
```

For bAbI Task 1, `source_text` is the supporting story and `application_prefix` contains the fixed bAbI instruction and the question. Candidate answers are the fixed location vocabulary seen in the selected split.

`full_context` must be rendered using the same instruction/question convention as `memory`; it differs only by exposing `source_text` in the axial model's normal token context.

## Axial model interface

The first backbone family is dense `Qwen3ForCausalLM` (start with `Qwen/Qwen3-1.7B`; repeat meaningful results on `Qwen/Qwen3-4B`).

The wrapper must expose:

```python
class AxialModel(Protocol):
    def encode_source(self, tokenized_source) -> SourceStates: ...
    def score_application(self, tokenized_application, memory_read=None) -> CandidateScores: ...
```

Implementation details may use Hugging Face hooks or a small Qwen block wrapper, but the training code must not depend directly on Qwen internals. The first reader injection points are selected upper Transformer blocks. Memory paths are initialized so that an empty memory yields the unmodified frozen-model behavior. An optional later interface ablation applies trainable rank-limited LoRA only to `q_proj` and `o_proj` of these same upper blocks. LoRA is disabled during source encoding and enabled only in the application pass; base Qwen weights remain frozen.

Qwen operates in no-thinking mode for v0. Evaluation supports `candidate_likelihood` (the primary diagnostic) and `free_generation` (greedy decoding with exact-candidate normalization). Both modes use the identical prompt and source-visibility variant, so output-format compliance can be measured separately from candidate likelihood.

## v0 writable embedding memory

`EmbeddingMemory` is a resettable collection of segments. A segment has trainable-format-independent runtime tensors:

```text
segment i = { key_i: [d_key], values_i: [n_value_tokens, d_value] }
```

The writer produces all segment keys and values from source hidden states. No sparse allocation, persistent state, nonlinear expert function, or runtime SGD is needed in v0. The first implementation uses one value vector per cell: learned write queries cross-attend to source states, then produce `N` per-example key/value pairs.

The reader receives application hidden states as queries and attends densely across the `N` cell keys/values. Its output is injected through a gated residual/cross-attention path into the final four axial layers. Gates are initialized near zero, so a fresh sidecar is close to frozen-Qwen behaviour. Later hierarchical segments may replace each value vector with several internal tokens without changing the source→state→reader boundary.

A later nonlinear-expert implementation must preserve this outward contract where possible: writer produces resettable state; reader accepts queries and returns memory features.

## Variants are protocol, not hidden switches

Every run declares one of these variants:

| Variant | Source in application context | Committed memory | Purpose |
|---|---:|---:|---|
| `no_source` | no | no | lower reference |
| `full_context` | yes | no | frozen-Qwen information upper reference |
| `memory` | no | yes | learned channel under test |
| `shuffled_memory` | no | another example's | causal negative control |
| `null_memory` | no | zero/initial | reader-injection control |

Future task-format adapters, LoRA, nonlinear experts, and multi-commit curricula must be named as new variants/configurations; they are not fallbacks inside v0.

## Checkpoint contract

A checkpoint stores:

- base model identifier and immutable revision/hash;
- tokenizer and fixed prompt template identifier;
- trainable state dict only (writer, reader, injection, global memory parameters);
- complete composed configuration;
- candidate vocabulary and dataset/split fingerprints;
- code revision; and
- metrics for every declared variant.

It never stores per-example reset memory state as learned global knowledge.
