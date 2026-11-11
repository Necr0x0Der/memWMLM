# TriviaQA oracle-context comparison

Date: 2026-11-11

Issue: https://github.com/Necr0x0Der/memWMLM/issues/11

## Result

Providing answer-containing TriviaQA evidence closed most, but not all, of the
closed-book gap between Qwen3-1.7B and Qwen3-4B. On the full 11,313-example
unfiltered `dev` split, oracle-context exact match was 63.44% for Qwen3-1.7B
and 68.06% for Qwen3-4B.

| Condition | Examples | Exact match | Token F1 | Correct |
|---|---:|---:|---:|---:|
| Qwen3-1.7B, closed book (#5) | 11,313 | 16.72% | 23.28% | 1,891 |
| Qwen3-4B, closed book (#5) | 11,313 | 30.39% | 37.52% | 3,438 |
| Qwen3-1.7B, oracle context | 11,313 | **63.44%** | **69.88%** | 7,177 |
| Qwen3-4B, oracle context | 11,313 | **68.06%** | **76.12%** | 7,700 |

Relative to the matched closed-book results, oracle context increased EM by
46.72 percentage points for 1.7B and 37.67 points for 4B. Qwen3-1.7B with
oracle context exceeded closed-book Qwen3-4B by 33.05 points.

The model-size gap changed from 13.67 points closed-book to 4.62 points with
oracle context, a reduction of 9.05 points, or approximately 66% of the
original gap. Thus access to explicit evidence accounts for a large portion of
the observed closed-book difference, but does not remove the reader-size
effect completely.

## Paired oracle comparison

The paired Qwen3-4B minus Qwen3-1.7B difference was:

- **+4.62 percentage points EM**, paired bootstrap 95% CI
  **[+3.87, +5.35]**;
- **+6.24 percentage points F1**, paired bootstrap 95% CI
  **[+5.60, +6.87]**.

Intervals used 10,000 paired resamples. EM used seed `20261111`; F1 used seed
`20261112`.

| Paired outcome | Questions |
|---|---:|
| Both correct | 6,519 |
| Only 4B correct | 1,181 |
| Only 1.7B correct | 658 |
| Both wrong | 2,955 |

The larger reader therefore gained 523 net exact matches even though both
models received identical passages for every question.

## Answer-present diagnostic

The accepted answer was detected in the supplied context for 10,014 questions
(88.52%). For the remaining 1,299 questions (11.48%), the post-hoc check found
no accepted answer in the supplied context; these cases necessarily used the
associated-evidence fallback. This is an answer-absence count, not necessarily
the exact fallback count, because a fallback chunk can contain an accepted
human-answer variant that was not used for oracle selection.

| Context diagnostic | Examples | 1.7B EM | 4B EM | 1.7B F1 | 4B F1 |
|---|---:|---:|---:|---:|---:|
| Accepted answer present | 10,014 | 71.58% | 76.54% | 77.31% | 83.44% |
| Accepted answer absent | 1,299 | 0.69% | 2.69% | 12.63% | 19.68% |

Among answer-present examples, the remaining 4B advantage was 4.96 points EM.
The very large separation between answer-present and answer-absent subsets
confirms that the aggregate gain depends on actually supplying usable answer
evidence. These subsets differ in difficulty, so their scores should not be
interpreted as a randomized causal comparison.

The configured maximum was five passages. Oracle selection returned between
one and five passages depending on selected or fallback passage availability:

| Passages supplied | Questions |
|---:|---:|
| 1 | 423 |
| 2 | 288 |
| 3 | 242 |
| 4 | 226 |
| 5 | 10,134 |

The generated summary files report `retrieval_top_k: 3` because the current
summary implementation reads the passage count from the first output record.
That field is not the configured maximum and should not be used; the run
metadata correctly records `top_k: 5`.

## Interpretation

The result supports a narrow conclusion: Qwen3-1.7B can use explicitly
provided factual evidence well enough to eliminate most of its closed-book
disadvantage to Qwen3-4B. The remaining matched-context gap indicates that
model size still affects reading, selecting, formatting, or otherwise using
the supplied evidence.

This is an oracle reader diagnostic, not an evaluation of ordinary RAG. Gold
`NormalizedAliases` were used to select answer-containing chunks, so the
experiment does not measure retrieval quality, generalization to an unrelated
corpus, or the native memWMLM memory architecture. It also does not establish
that the evidence caused every correct response: a model may already know some
answers from its weights.

## Output-contract diagnostics

| Diagnostic | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|
| Empty predictions | 0 | 0 |
| Multiline predictions | 4 | 1 |
| Reached 32-token generation cap | 73 | 277 |
| Mean input tokens | 1,809.6 | 1,809.6 |
| Median input tokens | 1,868 | 1,868 |
| 95th-percentile input tokens | 2,286 | 2,286 |
| Maximum input tokens | 21,127 | 21,127 |

Both models saw the same tokenized prompt lengths. The 4B model reached the
generation cap more often despite obtaining higher EM and F1; cap hits remain
ordinary scoring failures unless their normalized output matches an accepted
answer.

## Qualitative examples

These are the first examples in dataset order satisfying each category, rather
than examples selected for favorability.

| Category | ID | Question | Accepted answer | 1.7B | 4B |
|---|---|---|---|---|---|
| Both correct, answer present | `tc_40` | Who was the next British Prime Minister after Arthur Balfour? | Henry Campbell-Bannerman | Sir Henry Campbell-Bannerman | Sir Henry Campbell-Bannerman |
| Only 4B correct, answer present | `tc_33` | Which Lloyd Webber musical premiered in the US on 10th December 1993? | Sunset Boulevard | Phantom of the Opera | Sunset Boulevard |
| Only 1.7B correct, answer present | `tc_133` | What is the Japanese share index called? | Nikkei | Nikkei | Nikkei 225 |
| Both scored wrong, answer present | `tc_2` | Who was the man behind The Chipmunks? | David Seville | Ross Bagdasarian | Ross Bagdasarian |
| 4B correct, answer absent | `tc_13` | What star sign is Jamie Lee Curtis? | Scorpio | Libra | Scorpio |

`tc_2` exposes an alias limitation: Ross Bagdasarian created The Chipmunks and
performed as David Seville, but only `david seville` appears in the accepted
ground truths used by this run. Consequently, both semantically plausible
predictions receive zero EM. Conversely, `tc_133` shows exact-match strictness:
`Nikkei 225` does not exactly match the accepted `nikkei` alias.

## Protocol and provenance

- Dataset: official TriviaQA v1.0 unfiltered `dev`, all 11,313 examples.
- Dataset SHA-256:
  `720e25ee87c04444f73ab1cadad9d8d9fd3449c38e739a1a4f4ae2484f2b2009`.
- Oracle cache SHA-256 used by both models:
  `578ec8e927e3b87c54aed1d2ce163437c90f12243cf41037c55ccdc1a7b118db`.
- Oracle cache metadata SHA-256:
  `2657ab0b18a06b6c335593cec543405a5a69613b658bc71bfa94270ff5f9ac0b`.
- Models and requested tokenizer revisions:
  - `Qwen/Qwen3-1.7B` at
    `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
  - `Qwen/Qwen3-4B` at
    `1cfa9a7208912126459214e8b04321603b3df60c`.
- Repository commit used for both runs:
  `155fabfb7077a8e5e16f728784ea73f382ab8650`.
- Retrieval mode: `oracle_answer_containing`; gold was used for context
  selection; configured maximum `top_k=5`.
- Prompt version: `triviaqa_rag_short_answer_v1`.
- Qwen chat template with `enable_thinking=False`.
- Greedy decoding (`do_sample=false`), seed `0`, maximum 32 new tokens, batch
  size 1.
- Native BF16 weights, no quantization, device `cuda`.
- Maximum permitted input length: 30,000 tokens.
- PyTorch `2.15.0+cu132`; Transformers `5.22.2`.
- Primary scoring: TriviaQA normalized exact match; secondary scoring: maximum
  token F1 over accepted aliases.

The local-cache loader did not expose resolved model/tokenizer revisions in
metadata (`null`), so provenance relies on the immutable requested revisions
recorded above.

Output SHA-256 hashes:

- Qwen3-1.7B oracle JSONL:
  `ca4f36d454e2111fe5b2e25d163c56452a8e7b072feaf4d8368a0daa5a6e2d21`;
- Qwen3-4B oracle JSONL:
  `6101b0224d9e5e208a6e574ff16e8f9698eb6e6563f642ed1b2d21da28e34ca9`.

## What remains unproven

- Whether answer-independent retrieval can find similarly useful evidence.
- Whether a smaller model with a fixed external corpus can match 4B without
  gold-assisted passage selection.
- Whether the remaining reader gap reflects reasoning, instruction following,
  answer extraction, or response calibration.
- Whether the same result transfers to other factual datasets.
- Whether the behavior says anything about native memWMLM memory.
