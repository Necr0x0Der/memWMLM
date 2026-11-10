# Qwen3 size comparison on factual-knowledge datasets

Dates: 2026-11-07 to 2026-11-09

Issue: https://github.com/Necr0x0Der/memWMLM/issues/5

## TriviaQA result

On the complete TriviaQA v1.0 unfiltered `dev` split, Qwen3-4B substantially
outperformed Qwen3-1.7B under the matched no-context, no-thinking protocol.

| Dataset / domain | Examples | Model | Exact match | Token F1 | Correct |
|---|---:|---|---:|---:|---:|
| TriviaQA / `unfiltered-web` | 11,313 | Qwen3-1.7B | 16.72% | 23.28% | 1,891 |
| TriviaQA / `unfiltered-web` | 11,313 | Qwen3-4B | 30.39% | 37.52% | 3,438 |

The paired 4B − 1.7B difference was:

- **+13.67 percentage points EM**, paired bootstrap 95% CI
  **[+12.90, +14.47]**;
- **+14.24 percentage points F1**, paired bootstrap 95% CI
  **[+13.46, +15.01]**.

Bootstrap intervals used 10,000 paired resamples with seed `20261107`.

## Paired outcomes

| Outcome | Questions |
|---|---:|
| Both correct | 1,527 |
| Only 4B correct | 1,911 |
| Only 1.7B correct | 364 |
| Both wrong | 7,511 |

The larger model therefore gained 1,547 net exact matches, but did not dominate
on every item: the smaller model alone was correct on 364 questions.

## Protocol and provenance

- Dataset: official TriviaQA v1.0 unfiltered `dev`, all 11,313 examples.
- Dataset SHA-256:
  `720e25ee87c04444f73ab1cadad9d8d9fd3449c38e739a1a4f4ae2484f2b2009`.
- Models and tokenizer revisions:
  - `Qwen/Qwen3-1.7B` at
    `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
  - `Qwen/Qwen3-4B` at
    `1cfa9a7208912126459214e8b04321603b3df60c`.
- Repository commit used for both runs:
  `0f1eece5b66b08539f91f2e2b4d6cd7fb43cc83b`.
- Prompt version: `triviaqa_short_answer_v1`.
- Instruction: “Answer the factual question with only the shortest answer. Do
  not explain.”
- Qwen chat template with `enable_thinking=False`.
- Greedy decoding (`do_sample=false`), maximum 32 new tokens, batch size 1.
- Native BF16 weights, no quantization and no CPU offload.
- Inference seed: none required for deterministic greedy decoding.
- Primary scoring: official-style TriviaQA normalized exact match over aliases.
- Secondary scoring: maximum token F1 over aliases.
- PyTorch: `2.15.0+cu132`.

Output hashes:

- 1.7B JSONL:
  `133a57ec2c3d3c8d612431048cac5b0dbe4d22ca343993c771c967c0c313f052`;
- 4B JSONL:
  `5b00a12da4e9310979110fd064a04b00b12ba098dd04cc05e6d6dcc58be6e30d`.

## Output-contract diagnostics

| Diagnostic | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|
| Empty predictions | 0 | 0 |
| Multiline predictions | 8 | 8 |
| Reached 32-token generation cap | 84 | 26 |

Cap hits and multiline answers remain exact-match failures unless an official
alias matches after normalization. The lower cap-hit count for 4B suggests
better compliance with the requested short-answer format, but this experiment
does not separate factual recall from instruction-following effects.

## Qualitative error set

Examples below are the first item in dataset order satisfying each predeclared
category; they were not selected for favorability.

| Category | ID | Question | Accepted answer | 1.7B | 4B |
|---|---|---|---|---|---|
| 4B correct, 1.7B wrong | `tc_13` | What star sign is Jamie Lee Curtis? | Scorpio | Libra | Scorpio |
| 1.7B correct, 4B wrong | `tc_453` | In the Bible, who did the sun and moon stand still before? | Joshua | Joshua | King David |
| Both wrong, same answer | `tc_40` | Who was the next British Prime Minister after Arthur Balfour? | Henry Campbell-Bannerman | David Lloyd George | David Lloyd George |
| Both wrong, different answers | `tc_2` | Who was the man behind The Chipmunks? | David Seville | “no single man” | Bob Eubanks |
| 4B partial-F1 gain | `tc_56` | What claimed the life of singer Kathleen Ferrier? | cancer | Asthma | Lung cancer |
| 1.7B partial-F1 gain | `tc_69` | Rita Coolidge sang the title song for which Bond film? | Octopussy | “The Bond film is Skyfall” | The Living Daylights |

These examples show both additional factual recall in 4B and non-monotonic
errors. They also expose alias-metric behavior: “Lung cancer” receives partial
F1 rather than exact match against the broad `cancer` aliases.

## MMLU result

On all 14,042 questions from the MMLU `test` split, Qwen3-4B scored 59.91%
accuracy versus 26.21% for Qwen3-1.7B under the matched five-shot,
no-thinking, candidate-likelihood protocol.

| Dataset | Examples | Model | Accuracy | Correct |
|---|---:|---|---:|---:|
| MMLU `test` | 14,042 | Qwen3-1.7B | 26.21% | 3,681 |
| MMLU `test` | 14,042 | Qwen3-4B | 59.91% | 8,412 |

The paired 4B − 1.7B difference was **+33.69 percentage points**, with a
paired bootstrap 95% CI of **[+32.74, +34.63]**. The interval used 10,000
paired resamples with seed `20261109`.

### MMLU paired outcomes

| Outcome | Questions |
|---|---:|
| Both correct | 2,982 |
| Only 4B correct | 5,430 |
| Only 1.7B correct | 699 |
| Both wrong | 4,931 |

The larger model gained 4,731 net correct answers. Its advantage was positive
in every broad MMLU domain, but varied substantially by domain.

### MMLU domain accuracy

These are micro-averages over all questions assigned to each standard broad
domain.

| Domain | Examples | Qwen3-1.7B | Qwen3-4B | 4B − 1.7B |
|---|---:|---:|---:|---:|
| STEM | 3,018 | 22.90% | 58.65% | +35.75 pp |
| Humanities | 4,705 | 25.25% | 51.88% | +26.63 pp |
| Social sciences | 3,077 | 27.43% | 69.78% | +42.35 pp |
| Other (business, health, misc.) | 3,242 | 29.55% | 63.36% | +33.81 pp |

The largest broad-domain gain was in social sciences and the smallest was in
humanities. At subject level, all 57 subjects are recorded in the adjacent
`.summary.json` files. The largest and smallest observed gains were:

| Subject | Examples | Qwen3-1.7B | Qwen3-4B | Difference |
|---|---:|---:|---:|---:|
| High-school biology | 310 | 22.58% | 84.84% | +62.26 pp |
| High-school psychology | 545 | 25.32% | 83.49% | +58.17 pp |
| High-school computer science | 100 | 25.00% | 83.00% | +58.00 pp |
| High-school mathematics | 270 | 21.11% | 29.26% | +8.15 pp |
| Global facts | 100 | 18.00% | 27.00% | +9.00 pp |
| Professional law | 1,534 | 24.64% | 34.88% | +10.23 pp |

### MMLU answer-position diagnostic

The 1.7B result has a major calibration caveat. Gold answer letters were
approximately balanced, but Qwen3-1.7B selected `A` on 93.04% of questions.
Qwen3-4B was also position-skewed, though less extremely.

| Source | A | B | C | D |
|---|---:|---:|---:|---:|
| Gold answers | 22.95% | 24.65% | 25.51% | 26.89% |
| Qwen3-1.7B predictions | 93.04% | 0.17% | 0.58% | 6.21% |
| Qwen3-4B predictions | 40.24% | 10.06% | 17.70% | 32.00% |

Thus the measured size effect is real for this exact scoring protocol, but its
magnitude cannot be interpreted as a pure knowledge-capacity difference. It
also reflects model-dependent calibration or answer-position bias under direct
letter likelihood. The single-shuffle sensitivity control below tests whether
the size gap survives another option order; multiple permutations or calibrated
multiple-choice scoring would be needed to separate those effects more fully.

### MMLU shuffled-choice sensitivity control

To test sensitivity to the original option order without a fourfold evaluation,
the four test choices were deterministically shuffled once per question using
seed `20261109`. The correct letter was remapped to the same answer content;
five-shot demonstrations remained unchanged. Both models used the identical
generated dataset and permutation manifest.

| Model | Original accuracy | Shuffled accuracy | Change |
|---|---:|---:|---:|
| Qwen3-1.7B | 26.21% | 28.42% | +2.21 pp |
| Qwen3-4B | 59.91% | 59.08% | −0.83 pp |
| 4B − 1.7B | +33.69 pp | +30.66 pp | −3.03 pp |

The shuffled 4B − 1.7B difference was **+30.66 percentage points**, with a
paired bootstrap 95% CI of **[+29.73, +31.58]** from 10,000 resamples with seed
`20261109`.

| Shuffled paired outcome | Questions |
|---|---:|
| Both correct | 3,275 |
| Only 4B correct | 5,021 |
| Only 1.7B correct | 716 |
| Both wrong | 5,030 |

The shuffled gap remained positive in every broad domain:

| Domain | Examples | Qwen3-1.7B | Qwen3-4B | 4B − 1.7B |
|---|---:|---:|---:|---:|
| STEM | 3,018 | 26.84% | 56.83% | +29.99 pp |
| Humanities | 4,705 | 25.80% | 51.14% | +25.33 pp |
| Social sciences | 3,077 | 30.55% | 68.38% | +37.83 pp |
| Other (business, health, misc.) | 3,242 | 31.68% | 63.88% | +32.20 pp |

The shuffle balanced gold answer positions (`A`/`B`/`C`/`D` counts were
3,538/3,470/3,513/3,521), but it did not remove model letter preference.
Qwen3-1.7B still selected `A` on 92.82% of questions; Qwen3-4B selected `A` on
42.86%. Comparing selected answer content across the original and shuffled
runs, 1.7B chose the same content on only 27.24% of questions, versus 61.91%
for 4B. The 1.7B score is therefore highly order-sensitive at the item level,
even though its aggregate accuracy changed by only 2.21 points.

Seventy questions were conservatively flagged as containing possible explicit
position references such as “Both A and B”; shuffling can change their meaning.
Excluding them left the conclusion unchanged: shuffled accuracy was 28.48% for
1.7B and 59.16% for 4B, a +30.68-point gap.

Shuffled-run provenance:

- preparation algorithm:
  `sha256_sort_by_seed_example_id_and_original_index_v1`;
- permutation manifest SHA-256:
  `cccd1f4c8de19f34aa542ead9f100c0197547775b92354d722996dd7fc94e848`;
- repository commit used for both runs:
  `0ea19d4fff3286eee51115acbf32dc7d4514e170`;
- 1.7B shuffled JSONL SHA-256:
  `8b259a0c47bf31dcfc2fa3a350d9888ac89056b479e6f04abe94bf4c971366df`;
- 4B shuffled JSONL SHA-256:
  `7e9c8c590f13fe38d6c34e23a31bf7701ac8decd96d9cca477cc7d3fa1c5d3ae`.

This one-shuffle experiment is a sensitivity check, not a mathematical
debiasing procedure. It shows that the large size gap is not specific to the
original option order, while confirming that direct letter likelihood—most
severely for 1.7B—is not a clean measure of answer-content knowledge.

### MMLU qualitative error set

The four 4B-gain examples are the first such item in dataset order within each
broad domain. The reversal is the first item where only 1.7B was correct. This
deterministic rule avoids selecting examples for favorability.

| Category | ID | Question | Correct option | 1.7B | 4B |
|---|---|---|---|---|---|
| 4B gain / STEM | `abstract_algebra:test:2` | Find all zeros of `x^5 + 3x^3 + x^2 + 2x` in `Z_5`. | D: 0,4 | A: 0 | D: 0,4 |
| 4B gain / humanities | `formal_logic:test:0` | Identify the conclusion of Freud's argument about consciousness and unconscious complexes. | D: It is difficult to convince people of the unconscious… | A: It is hard not to verify… | D: It is difficult to convince people of the unconscious… |
| 4B gain / social sciences | `econometrics:test:1` | Main difference between Dickey–Fuller and Phillips–Perron unit-root tests? | C: PP corrects for autocorrelated residuals | A: single-equation vs systems approach | C: PP corrects for autocorrelated residuals |
| 4B gain / other | `anatomy:test:0` | Effect of facial-nerve compression at the stylomastoid foramen? | A: paralysis of facial muscles | D: paralysis plus taste/lacrimation/salivation loss | A: paralysis of facial muscles |
| 1.7B-only reversal | `abstract_algebra:test:5` | Two statements about counts of group elements of order 15. | A: True, True | A: True, True | D: False, True |

### MMLU protocol and provenance

- Dataset: `cais/mmlu`, revision
  `c30699e8356da336a370243923dbaf21066bb9fe`, complete `test` split.
- Original archive SHA-256:
  `bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b`;
  per-file hashes are preserved in each run's metadata.
- Models and tokenizers used the same immutable revisions as TriviaQA.
- Repository commit used for both runs:
  `8df6870067025c8bb0a2e9a00f35c678ef9428f1`.
- Prompt version: `mmlu_official_5shot_chat_v1`; five fixed demonstrations
  from the official `dev` split for each subject.
- Qwen chat template with `enable_thinking=False`.
- Scoring: summed conditional log likelihood of the candidate answer letters
  `A`, `B`, `C`, and `D`; highest-scoring letter selected.
- Seed `0`; no stochastic sampling; batch size 1.
- Native BF16 weights, no quantization and no CPU offload.
- PyTorch `2.15.0+cu132`; Transformers `5.22.2`.

Output hashes:

- 1.7B JSONL:
  `97e79b28dadaaa44235539ad7bf2f6185aa6aa50bced99e923205d8c270fe76e`;
- 4B JSONL:
  `05ef0088853ad6257edd87e929334a12bd61b4af870cbcdfec17c304db69cb08`.

## What this establishes

Across both TriviaQA and MMLU, Qwen3-4B clearly outperformed Qwen3-1.7B under
matched prompts, precision, and model-family conditions. The TriviaQA gain was
+13.67 pp exact match; the MMLU gain was +33.69 pp accuracy and was positive in
all four broad domains. This is consistent with greater capacity improving
factual recall and broad task performance.

The evidence still has important limits:

- MMLU mixes factual knowledge with reasoning and exhibits strong
  model-dependent answer-position bias in this candidate-letter protocol;
- TriviaQA does not provide semantic domain labels, so its gain cannot be
  decomposed by domain;
- benchmark contamination cannot be ruled out for either dataset;
- no 8B model, RAG condition, retrieval control, multi-permutation average, or
  calibrated multiple-choice evaluation was performed;
- the experiment cannot isolate stored factual knowledge from better reasoning,
  instruction following, or retrieval within model weights.

The robust conclusion is therefore that 4B performs better than 1.7B on both
evaluated benchmarks, and its MMLU advantage persists under one deterministic
random option order. The improvement across every MMLU domain does not reveal
a clean domain-specific modular pattern, and these knowledge-heavy benchmarks
do not cleanly separate factual capacity from reasoning ability.

This result should also be read with the earlier frozen full-context bAbI
comparison recorded in [`docs/decisions.md`](../../docs/decisions.md). There,
4B improved substantially on only one of six tasks (QA4), improved slightly on
QA12, regressed slightly on QA11, and was essentially unchanged on the other
tasks. Those are single reported evaluations rather than matched multi-seed
controls, but they show little evidence of a broad reasoning gain from scale.
The contrast between limited bAbI gains and large TriviaQA/MMLU gains remains
compatible with part of the larger model's advantage being externalizable
factual knowledge; it does not identify or prove such modules.

The current benchmark design therefore fails to isolate domain-specific factual
modules. A direct follow-up is to test whether Qwen3-1.7B with a fixed external
corpus and matched RAG protocol can close the TriviaQA gap to closed-book 4B,
rather than continuing to infer modularity from model-size deltas alone.
