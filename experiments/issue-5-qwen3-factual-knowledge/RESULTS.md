# Qwen3 size comparison on TriviaQA

Date: 2026-11-07

Issue: https://github.com/Necr0x0Der/memWMLM/issues/5

## Result

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

## What this establishes

For TriviaQA open-domain factual recall, 4B is clearly stronger than 1.7B under
the matched protocol. The result is consistent with additional model capacity
carrying useful factual knowledge and/or improving short-answer compliance.

It does **not** establish the issue's broader domain-specific claim:

- only one dataset was evaluated;
- TriviaQA labels the whole split as `unfiltered-web` and provides no semantic
  domain labels, so finer per-domain accuracy cannot be reported faithfully;
- no MMLU run is present, so there is no multiple-choice or reasoning control;
- no 8B model was evaluated;
- benchmark contamination cannot be ruled out;
- the experiment does not distinguish factual knowledge from better retrieval
  within weights or better instruction following.

Accordingly, this is a strong single-dataset size effect, not yet evidence that
gains are concentrated in particular factual domains or separable from general
reasoning.
