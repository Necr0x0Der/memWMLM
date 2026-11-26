# Delta-rule memory on TriviaQA: existing heads vs trained interface

Dates: 2026-11-10 to 2026-11-18 (corrections and checks 2026-11-17/18);
follow-up checks and small-bank training 2026-11-19 to 2026-11-25 (section
"Follow-up: small banks with training")

Issue: https://github.com/Necr0x0Der/memWMLM/issues/6

## Answer

> **Update 2026-11-25 (follow-up, small banks).** The corpus-scale answer
> below stands. On small banks (1–64 passages per memory, Qwen3-1.7B,
> 1,024 eval questions) the follow-up found that the memory *can* be made
> to work with training, for both variants:
>
> - **Variant 1 + application-side LoRA + learned per-token β** (trained on
>   TriviaQA train questions, disjoint from eval): memory EM 54.1 / 41.4 /
>   26.5 at N = 1 / 4 / 16 vs 20.2 closed-book (same LoRA), far above
>   shuffled and answer-swapped memories and above a control LoRA trained
>   with unrelated memory. With 4,096-wide keys it still retrieves at
>   N = 64 (18.9 vs 15.7 shuffled) but no longer beats closed-book.
> - **Variant 2, step B** (writer = Qwen's own k/v projections + rank-8
>   LoRA, reader = q/o LoRA, trained on WikiText fill-in-the-blank only):
>   memory 48.5 / 25.0 / 10.5 / 4.0 at N = 1 / 4 / 16 / 64, above shuffled
>   and answer-swapped memories at every N; +31 / +8 over plain Qwen's
>   closed-book (17.1) at N = 1 / 4. Its reader lowers closed-book to 7.8.
> - Passages pasted **in context** still beat every memory by 13–49 points
>   (LoRA: 67.3 at N = 1, 64.7 at N = 64), and the gap grows with N.
>
> Untrained (frozen) Variant 1 stays negative in every follow-up check.

**The claim is refuted for both variants on this setup.** A write-once
delta-rule memory over a 4,342-passage TriviaQA corpus does not let frozen
Qwen3-1.7B answer held-out questions better than closed-book, let alone
approach RAG (BM25@1: +34.2 points over closed-book on the same questions).

- **Variant 1 (existing Qwen heads)** carries a small, answer-dependent
  signal for 1–4 written passages (shrinking); it is gone at 16 passages and
  absent at corpus scale, with or without adaptation (trained gates, LoRA).
  Adaptation is required and does not suffice: LoRA's gain is plain
  fine-tuning (identical with an empty memory).
- **Variant 2 (new writer/reader trained on general text)** never learned to
  retrieve: at all four memory sizes the trained interface performs the same
  with the right memory as with a memory of unrelated text. After a series of
  pre-declared development fixes it stores a little content for 1–4
  sentences, nothing at 64.
- **Expanded keys (Variant 1 with key size D ∈ {512…4096}):** a bigger
  memory does not help at corpus scale. Hebbian stays at closed-book level
  (dev_in 18.6–19.3 vs 18.2, identical with an answer-swapped memory); the
  delta rule stays harmful (4.1–8.2). *Corrected 2026-11-18:* this does not
  show that capacity is not the bottleneck — 184k written tokens far exceed
  any tested key size. A direct small-bank test (section "Corrections and
  capacity test") found that 32× more key space does not extend the content
  signal to more passages (both models), though it makes the read much less
  harmful. So the frozen read itself is limiting; capacity is not ruled out
  at corpus scale. A rerun of D 4,096 on the full memory with both fixes
  is still below closed-book and equal to the answer-swapped memory (except
  1.7B delta, +1.2 over swapped but −9.8 vs closed-book).
- **Qwen3-4B (optional check)** gives the same answer at corpus scale:
  closed-book 35.2, BM25@1 60.5, and no frozen, adapted, expanded-key or
  V2 (d 512 … 4096) memory beats its own answer-swapped memory (LoRA's +1.8 is again matched with `S = 0`). Two
  things differ from 1.7B: (1) on small banks the frozen answer-dependent
  signal is stronger at N = 1 (+10 vs shuffled); the first run also met the
  pre-registered criterion at 16 passages, but that did **not** replicate
  with the first-token fix (2026-11-18: signal at N = 1 only), and the read
  lowers EM from 33.8 to ~5 at N = 16; (2) the final V2b configuration
  passes its WikiText decision rule, but on TriviaQA it falls to 8.4 vs 35.2
  closed-book, no better than its answer-swapped memory.

In no condition does the model reproduce a swapped (counterfactual) answer
from memory (follow ≤ 0.1% everywhere, both models; 25–32% when the same
swapped passage is in context).

## Setup

- Model: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
  frozen, BF16, chat template with `enable_thinking=False`, greedy, ≤ 32 new
  tokens. Prompt `triviaqa_short_answer_v1` (as issue #5).
- Data: TriviaQA RC v1.0 (`artifacts/triviaqa/`, gitignored), one
  answer-bearing evidence sentence window per question as its passage
  (`prepare_banks.py`), alias-disjoint train/dev construction
  (`prepare_global.py`, manifest `runs/delta-rule/global-v1/manifest.json`).
- **Global memory (primary):** one memory written once from 4,342 passages
  (183,667 tokens; shuffled order, seed 20261113), then all questions asked
  against that fixed memory without passages in context. Question groups
  (512 each): `train_seen` (passage written, question used for V1
  adaptation), `train_heldout` (passage written, never trained on), `dev_in`
  (dev question, passage written), `dev_out` (dev question, no passage
  written). dev_in vs closed-book and vs BM25@1 is the primary comparison;
  dev_out measures harm.
- Controls: null memory (`S = 0`, equals closed-book for frozen readers),
  shuffled memory (small banks), counterfactual memory (each dev_in passage
  with its answer swapped for another question's answer; "follow" = model
  outputs the swapped-in answer), novel questions (`train_heldout`,
  `dev_out`), memory trained with `S = 0` (fine-tuning control), and the
  known/unknown split below (contamination).
- Protocol: fixed in `README.md` before each run; every later change is a
  dated amendment there. Raw logs in `NOTES.md`.

## Baselines (dev_in, 512 questions)

| Condition | EM % | Δ vs closed-book [95% CI] |
|---|---:|---|
| closed-book | 18.2 | — |
| BM25@1 over the written corpus, passage in context | 52.3 | +34.2 [+29.3, +39.3] |

Small banks (1,024 dev questions, `banks-v2`): oracle passage in context
70.9; full context 68.4 / 62.3 / 53.8 at N = 4 / 16 / 64; BM25@1 67.5 /
66.9 / 65.1. With question-blind passage selection (`banks-first-v1`)
BM25@1 drops to 54.9 / 51.3 / 47.7 (oracle 67.9), so the overlap selection
favours BM25.

## Variant 1 — delta rule on existing Qwen heads

Pre-RoPE `k_norm`ed keys and values of every KV head (28 layers × 8 heads)
written into a 128 × 128 state per head; every query head reads its group's
state with its pre-RoPE query; `gate · read` is added to the attention
output before `o_proj`. Keys are centred with a calibration mean.

### Small banks (frozen, no training)

1,024 dev questions; closed-book 17.1. EM %:

| Config | N = 1 | 2 | 4 | 16 |
|---|---:|---:|---:|---:|
| delta β=α=1, g 0.2 (pre-registered) | **21.6** | 18.8 | 13.3 | 0.4 |
| shuffled memory | 15.7 | 13.4 | 10.5 | 0.9 |
| counterfactual memory | 17.4 | 14.6 | 11.0 | 0.4 |
| delta β 0.5, α 0.99, g 0.5 (tuned on calibration) | **25.1** | 15.5 | 10.3 | 6.0 |
| Hebbian β = 1/T, g 4 (tuned on calibration) | 23.4 | 19.1 | 17.3 | 17.0 |

At N = 1 the pre-registered memory beats closed-book by +4.5 [+2.4, +6.5],
shuffled by +5.9 [+3.8, +7.9] and counterfactual by +4.2 [+2.4, +6.0];
counterfactual follow 0% (in context 32.2%). The signal is answer-dependent
but tiny (~6% of the oracle-in-context gain) and shrinks with N: vs
shuffled / counterfactual memory +5.5 / +4.3 at N = 2, +2.7 [+1.0, +4.5] /
+2.2 [+0.5, +4.0] at N = 4, −0.5 [−1.2, +0.1] / 0.0 [−0.4, +0.4] at N = 16
(paired bootstrap, seed 20261111). An earlier draft said "gone by N = 4";
that was wrong, the N = 4 difference is small but excludes 0.

### Global memory, frozen and adapted (seeds 0/1/2)

Adaptation trained on TriviaQA-train questions (`train_seen` +
`train_other`, 3,318 questions, 2 epochs, batch 16 = 416 steps) against the
global memory. dev_in EM %:

| Stage | rule | memory (s0/s1/s2) | S = 0 (s0/s1/s2) | counterfactual |
|---|---|---|---|---|
| frozen | delta β 0.5 | 0.0 | 18.2 | 0.0 |
| frozen | Hebbian β 1/T, g 2 / g 4 | 17.2 / 16.8 | 18.2 | 17.4 / 16.6 |
| gates (448 params) | delta | 0 / 0 / 0 | 18.2 | 0 |
| gates | Hebbian | 19.9 / 20.1 / 20.1 | 18.2 | within 0.4 |
| LoRA r8 on q/o (1.84 M) | delta | 9.2 / 10.6 / 9.0 | 20.5 / 19.1 / 19.5 | within 0.8 |
| LoRA r8 | Hebbian | 24.6 / 25.6 / 25.8 | 24.8 / 24.6 / 25.8 | within 0.4 |
| LoRA r8 trained with S = 0 | — | — | 24.4 / 24.6 / 26.0 | — |

Paired dev_in CIs (seed 0): LoRA Hebbian vs closed-book +6.4 [+3.3, +9.8],
its own S = 0 +6.6 [+3.5, +9.8]; vs counterfactual memory −0.2 [−0.6, 0.0].
Gate Hebbian +1.8 [0.0, +3.5], vs counterfactual 0.0. The delta state at
corpus scale is ill-conditioned (mean Frobenius norm 1,351 per head vs 5.5
Hebbian) and its read destroys generation.

**Adaptation:** required (frozen global delta reads 0%), and gates or LoRA do
not make the memory carry facts: every gain is identical with an empty or
answer-swapped memory, and identical for dev_in and dev_out.

### Expanded keys (capacity test)

Same frozen Variant 1, but each head's 128-dim key (and query) is expanded
with DPFP (`[relu(x), relu(−x)]` then products with ν rolled copies) to
D = 256ν ∈ {512, 1024, 2048, 4096}, so each head's state is D × 128 instead
of 128 × 128 (4–32× the memory). No training. The gate is chosen per run by
`train_seen` EM only (README amendment); dev groups untouched by selection.
EM %, 512 questions per group:

| rule, D | gate | train_seen | train_heldout | dev_in | dev_out | cf dev_in (follow) |
|---|---|---:|---:|---:|---:|---|
| closed-book | — | 21.9 | 25.6 | 18.2 | 16.0 | — |
| delta β 0.5, 512 | 0.1 | 8.4 | 9.8 | 8.2 | 7.0 | 8.2 (0) |
| delta, 1024 | 0.1 | 4.5 | 5.7 | 4.1 | 5.1 | 4.5 (0) |
| delta, 2048 | 0.1 | 10.7 | 13.5 | 7.8 | 10.2 | 8.4 (0) |
| delta, 4096 | 0.1 | 10.2 | 10.9 | 6.6 | 9.0 | 6.0 (0) |
| Hebbian β 1/T, 512 | 2 | 21.1 | 23.8 | 19.3 | 16.8 | 19.5 (0) |
| Hebbian, 1024 | 2 | 21.7 | 24.8 | 18.8 | 17.2 | 18.6 (0) |
| Hebbian, 2048 | 2 | 21.5 | 24.2 | 18.6 | 17.2 | 18.6 (0) |
| Hebbian, 4096 | 2 | 20.9 | 24.4 | 18.8 | 17.2 | 18.9 (0) |

Paired dev_in CIs: Hebbian D 512 vs closed-book +1.2 [−0.6, +2.9], vs
counterfactual −0.2 [−0.6, 0.0]; Hebbian D 4096 +0.6 [−1.2, +2.3] / −0.2
[−1.0, +0.4]; delta D 2048 −10.4 [−13.5, −7.2] / −0.6 [−1.8, +0.6].

Larger D does not move any group, dev_in (passage written) is no better than
dev_out (not written), and the answer-swapped memory scores the same. For the
delta rule every gate above the smallest (0.1) gives 0% and EM falls
monotonically with the gate, so a smaller gate would only approach
closed-book. Expanded keys therefore do not change the Variant 1 answer at
corpus scale. (*2026-11-18:* this run centred keys but not queries before
DPFP, and the corpus exceeds every tested key size, so it is not a capacity
test; see "Corrections and capacity test". The D 4,096 row is superseded
by "Global expanded-key rerun with both fixes".)

## Variant 2 — trained writer/reader on general text

New `DeltaInterface` at decoder layer 14: writer `k = norm(W_K x)`,
`v = W_V x`, per-token `β = σ(w_β x)`, exact delta rule into one `d × d`
state; reader `q = norm(W_Q x)`, layer output `h + W_O (qᵀS)`, `W_O`
zero-initialised. Qwen frozen. Trained only on WikiText-103 sentence cloze
(120k train sentences; answer-leakage filter against all TriviaQA aliases of
the corpus): each step writes N random sentences into a fresh S and asks 32
fill-in-the-blank questions about them; 2,000 steps × 32 queries = 64,000
presented queries (~2.4 M sentence writes). Nothing tuned on TriviaQA.

### Pre-registered run (seed 0)

| d | params | dev_in | dev_out | cf dev_in (follow) | dev_in at corpus 512 / 1,024 / 2,048 | WikiText cloze N 1,024 / closed |
|---|---:|---:|---:|---|---|---|
| 512 | 4.2 M | 17.8 | 15.4 | 17.6 (0) | 17.0 / 17.2 / 17.4 | 2.7 / 1.4 |
| 1024 | 8.4 M | 18.4 | 17.0 | 18.8 (0) | 17.8 / 18.4 / 18.0 | 2.7 / 1.4 |
| 2048 | 16.8 M | 17.6 | 15.8 | 18.0 (0) | 17.4 / 17.2 / 17.4 | 2.9 / 1.4 |
| 4096 | 33.6 M | 15.6 | 16.0 | 15.6 (0) | 15.2 / 15.2 / 15.2 | 2.5 / 1.4 |

Closed-book: dev_in 18.2, dev_out 16.0. dev_in vs closed-book: −0.4
[−3.3, +2.5] / +0.2 [−2.7, +2.9] / −0.6 [−3.5, +2.3] / −2.5 [−5.3, +0.2];
d = 4096 vs its counterfactual memory 0.0 [−1.2, +1.0]. d = 4096 was trained
with write chunk 512 (chunk 1024 ran out of GPU memory at step ~575; the
chunked delta rule is exact for any chunk size).

**Why it fails — the positive control.** Teacher-forced WikiText loss of
the d = 2048 interface with the *right* memory vs a memory of *other*
sentences: 7.06 vs 7.04 (N = 1), 2.96 vs 2.97 (N = 512). The trained
interface reduces loss from ~7 to ~3 with *any* memory: it learned a
content-independent prior of the cloze answers, not retrieval.

### Development (V2b, WikiText only, pre-declared decision rule)

Diagnosed and addressed in turn (each a README amendment; details in NOTES):
key and value from the same token (→ key t, value t+1); large banks only (→
N ∈ {1, 4, 16}); generic-bias shortcut (→ learned constant state S0);
Qwen layer-14 massive activation (dim 1999, mean |x| 25 vs median 0.44;
keys' mean pairwise cosine 0.46 → 0.65 after training; → fixed per-dimension
standardisation, cosine 0.01); answer-format confound (→ 3 worked examples
as chat turns). Final pilot (d = 1024, 256 queries):

| N | loss real / wrong / none | EM real / wrong |
|---|---|---|
| 1 | 3.39 / 3.61 / 7.03 | 1.6 / 1.2 |
| 4 | 3.13 / 3.22 / 6.72 | 3.1 / 2.0 |
| 64 | 3.40 / 3.39 / 6.89 | 1.6 / 1.6 |
| 1,024 | 3.32 / 3.26 / 6.73 | 0.8 / 1.6 |

Decision rule (fixed before the pilot): proceed only if the loss gap at
N = 1 is ≥ 0.1 and real EM > wrong EM at N = 64. The first holds (0.23), the
second does not, so **Variant 2 is reported negative** and was not evaluated
again on TriviaQA.

## Qwen3-4B (optional check)

`Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`, same data,
prompt and scoring; plan and success criterion fixed in the README amendment
before any 4B result (`run_4b.sh`, outputs `runs/delta-rule/qwen3-4b/`).
Adapted runs and V2b used token-weighted micro-batches to fit 24 GB (README
notes). The criterion: 4B changes this issue's answer only if a memory beats
on dev_in both closed-book (or its own `S = 0`) and its answer-swapped memory,
CIs excluding 0; or, on small banks, memory beats both shuffled and
counterfactual memory at N ≥ 4.

**Global memory, dev_in EM %** (512 questions; seeds 0/1/2 for adapted runs;
CIs seed 0, vs 4B closed-book):

| Condition | memory | S = 0 | counterfactual (follow) | vs closed-book | vs counterfactual |
|---|---|---|---|---|---|
| closed-book | 35.2 | — | — | — | — |
| BM25@1 | 60.5 | — | — | +25.4 [+20.5, +30.3] | — |
| frozen delta β 0.5, g 0.1 (by train_seen) | 0.4 | 35.2 | 0.2 (0) | | |
| frozen Hebbian, g 2 (by train_seen) | 34.6 | 35.2 | 34.6 (0) | | |
| gates (1,152) delta | 28.3 / 19.0 / 18.8 | 35.2 | 28.7 / 19.7 / 19.5 (0) | −6.8 [−10.7, −2.9] | −0.4 [−1.6, +0.6] |
| gates Hebbian | 35.2 / 35.4 / 35.4 | 35.2 | 35.0 / 35.4 / 35.2 (0) | 0.0 [−2.1, +2.1] | +0.2 [0.0, +0.6] |
| LoRA r8 (3.84 M) delta | 25.6 / 23.4 / 32.4 | 35.5 / 35.2 / 36.1 | 25.6 / 23.0 / 31.8 (0) | −9.6 [−13.7, −5.5] | 0.0 |
| LoRA r8 Hebbian | 36.9 / 39.1 / 36.5 | 37.5 / 40.4 / 37.5 | 36.7 / 39.3 / 36.7 (0) | +1.8 [−1.8, +5.3] | +0.2 [−0.4, +1.0] |
| LoRA r8 trained with S = 0 | 0 / 0 / 0 | 37.3 / 38.9 / 37.7 | 0 (0) | S = 0: +2.1 [−1.2, +5.7] | — |
| V2 pre-registered, d 512, layer 18 | 31.5 | 35.2 | 31.5 (0) | −3.7 [−7.4, −0.2] | 0.0 [−0.6, +0.6] |
| V2 pre-registered, d 1024, layer 18 | 31.5 | 35.2 | 30.9 (0) | −3.7 [−7.2, −0.2] | +0.6 [−0.6, +1.8] |
| V2 pre-registered, d 2048, layer 18 | 28.1 | 35.2 | 27.9 (0) | −7.0 [−10.5, −3.7] | +0.2 [−0.8, +1.2] |
| V2 pre-registered, d 4096, layer 18 | 20.1 | 35.2 | 19.7 (0) | −15.0 [−19.1, −10.9] | +0.4 [−0.4, +1.2] |
| V2b final config (passed WikiText rule) | 8.4 | 35.2 | 9.8 (0) | −26.8 [−31.1, −22.7] | −1.4 [−2.9, 0.0] |

dev_out closed-book 32.4; no condition gains more on dev_in (passage written) than on
dev_out (not written); e.g. LoRA Hebbian dev_out 37.3 / 39.1 / 37.5 (NOTES).
The frozen gate sweep repeats 1.7B: delta reads 0% above g 0.1, Hebbian
train_seen 38.5 / 39.5 / 38.9 / 24.2 at g 1 / 2 / 4 / 8 vs closed-book 39.7.

V2 widths d 512 / 2,048 / 4,096 were added on 2026-11-17 (README amendment
"4B coverage"; d 4,096 trained with micro-batches and write chunk 2,048 to
fit memory, an exact change). As at 1.7B, width does not help: the wider
the interface, the lower dev_in, and right = answer-swapped memory at every
width. Corpus-size curve (dev_in at 512 / 1,024 / 2,048 / 4,342 stored
passages): d 512 31.5 / 31.1 / 31.6 / 31.5; d 2,048 31.1 / 29.5 / 29.3 /
28.1; d 4,096 29.9 / 27.9 / 23.2 / 20.1. Probe (WikiText val, loss real /
unrelated memory): no width separates them (largest gap d 4,096 at N 64,
4.07 / 4.34, EM 3.9 / 3.1 of 256; at N 1 all within 0.03).

**V1 expanded keys, 4B** (frozen heads, DPFP keys of size D, gate chosen on
train_seen, ties → smaller; 2026-11-17 amendment). EM %:

| run | gate | train_seen | train_heldout | dev_in | dev_out | cf dev_in (follow) |
|---|---|---|---|---|---|---|
| closed-book | — | 39.7 | 40.6 | 35.2 | 32.4 | — |
| delta D 512 | 0.1 | 31.8 | 29.7 | 24.6 | 22.7 | 24.0 (0) |
| delta D 1,024 | 0.1 | 18.9 | 17.8 | 14.1 | 13.9 | 14.4 (0) |
| delta D 2,048 | 0.1 | 30.1 | 29.9 | 23.6 | 20.3 | 23.2 (0) |
| delta D 4,096 | 0.1 | 31.8 | 30.9 | 25.0 | 23.4 | 25.2 (0) |
| Hebbian D 512 | 2 | 38.9 | 38.1 | 31.1 | 32.4 | 30.5 (0) |
| Hebbian D 1,024 | 2 | 37.7 | 37.9 | 32.8 | 33.0 | 33.0 (0) |
| Hebbian D 2,048 | 2 | 38.1 | 39.3 | 33.6 | 33.8 | 34.0 (0) |
| Hebbian D 4,096 | 2 | 38.7 | 38.9 | 34.0 | 34.2 | 34.2 (0) |

dev_in CIs: delta D 4,096 vs closed-book −10.2 [−13.5, −6.8], vs
counterfactual −0.2 [−1.2, +0.6]; Hebbian D 4,096 −1.2 [−3.5, +1.0] and
−0.2 [−1.0, +0.6]. Both selected gates sit at the low edge of their grids
(as at 1.7B for delta); per the amendment this is reported, not re-tuned —
lower gates move toward `S = 0` = closed-book. Same picture as 1.7B: a
bigger key space does not make the frozen heads read written facts (these
runs used uncentred queries; see the 2026-11-18 capacity test, and the
fixed D 4,096 rerun under "Global expanded-key rerun with both fixes").

**Small banks** (banks-v2, 1,024 dev questions, frozen delta β = α = 1, g 0.2,
centred keys, as pre-registered for 1.7B). EM %; closed-book 33.8, oracle
passage in context 74.7, counterfactual passage in context 21.3 (follow 24.7):

| N | memory | shuffled | counterfactual (follow) | memory − shuffled | memory − counterfactual |
|---|---:|---:|---|---|---|
| 1 | 37.8 | 26.8 | 29.2 (0.1) | +11.0 [+8.6, +13.5] | +8.6 [+6.7, +10.5] |
| 4 | 18.4 | 16.3 | 15.7 (0) | +2.1 [−0.2, +4.3] | +2.6 [+1.1, +4.3] |
| 16 | 6.7 | 4.9 | 4.2 (0) | +1.9 [+0.1, +3.6] | +2.5 [+1.3, +3.9] |

1.7B for comparison (same test): memory − shuffled / − counterfactual +2.7 /
+2.2 at N = 4 (both CIs exclude 0), −0.5 / 0.0 at N = 16.

**Variant 2 positive control** (`probe_v2.py`, WikiText val, loss right /
unrelated memory; EM right / unrelated):

| N | V2 pre-registered | V2b final config |
|---|---|---|
| 1 | 6.88 / 6.85; 1.6 / 1.6 | 3.06 / 3.26; 5.9 / 3.9 |
| 4 | 5.55 / 5.63; 4.3 / 4.3 | 2.86 / 2.92; 5.5 / 4.7 |
| 64 | 2.83 / 2.81; 4.7 / 4.3 | 2.99 / 3.02; 4.7 / 3.5 |
| 1,024 | 3.04 / 3.04; 7.0 / 8.6 | 3.95 / 3.67; 2.3 / 3.1 |

V2b passes the pre-declared rule (gap 0.20 ≥ 0.1 at N = 1; real EM > wrong EM
at N = 64, by 3 of 256 queries) and was therefore evaluated on TriviaQA
(table above): dev_in falls with corpus size (17.6 / 11.3 / 6.1 at 512 /
1,024 / 2,048 passages, 8.4 at 4,342), and its WikiText cloze EM with 1,024+
stored sentences (1.0) is below no memory (2.3). It was trained on N ≤ 16
only, so it does not hold up at the scale it is used at.

**Verdict against the pre-registered criterion.** Global memory: not met by
any condition (including the 2026-11-17 V2 widths and expanded keys). Small banks: met at N = 16 (not at N = 4, where the shuffled
CI includes 0) in this run; *not replicated* with the 2026-11-18 first-token
fix at the same gate (N = 16: +0.3 [−1.3, +2.0] vs shuffled), so treat it
as not robust. Read together: a larger model keeps a small answer-dependent
trace of a few written passages longer, but the same read costs far more EM
than it adds, and nothing of it survives at corpus scale. The answer to the
issue does not change; the small-bank part is a 4B difference worth noting.

## Corrections and capacity test (2026-11-18)

An independent code review (no access to these conclusions) found no bug
that makes memory content irrelevant: the answer-swapped passages differ
from the originals (512/512 global, 1,024/1,024 banks-v2), the state is read
on every decoding step, and GQA head mapping, k⊗v layout, padding masks,
cache keys, V2 label alignment and scoring are correct. It found three
problems, recorded in a README amendment before the test below:

1. **Dropped first token.** Qwen3 adds no BOS, so skipping the "sink"
   position dropped the first content token of every written passage (the
   answer starts there in 61 of 1,024 eval passages). Fixed with
   `--sink-prefix` (an `<|endoftext|>` in front, which is what gets
   skipped). Earlier runs are not rerun. Check that this does not hide an
   effect (`check_first_token.py`, output in `records/runs.jsonl`, path `check_first_token.out`):
   on the 483 of 512 dev_in questions whose passage does not start with the
   answer (answer fully written), memory − answer-swapped memory is within
   [−1.2, +0.6] for every global run (both models, all V1 and V2
   conditions), the same as on all 512. In WikiText (V2 training) the answer
   starts the sentence in 15 of 4,596 val rows.
2. **Uncentred queries with expanded keys.** Keys were centred before DPFP,
   queries were not. Fixed with `--center-queries`. Earlier expanded-key
   runs used the mismatch.
3. **"Capacity is not the bottleneck" was overstated** (the corpus far
   exceeds every tested key size). Corrected above.

**Capacity test** (banks-v2, 1,024 eval questions, frozen V1, delta
β = α = 1, centred keys, sink fix; A = plain keys D 128, B = DPFP D 4,096
with centred queries; gate chosen on the calibration split by memory −
shuffled at N = 16). Gates: 1.7B A 0.05 (grid edge), B 0.1; 4B A 0.2,
B 0.1. EM %; differences with paired 95% CIs; follow ≤ 0.1 everywhere:

| model, config | N | memory | shuffled | cf | memory − shuffled | memory − cf |
|---|---:|---:|---:|---:|---|---|
| 1.7B A (closed 17.1) | 1 | 18.0 | 16.9 | 17.2 | +1.1 [0.0, +2.1] | +0.8 [0.0, +1.7] |
| | 4 | 17.9 | 17.0 | 17.9 | +0.9 [−0.4, +2.1] | 0.0 [−0.8, +0.8] |
| | 16 | 17.1 | 17.5 | 17.6 | −0.4 [−1.8, +1.0] | −0.5 [−1.5, +0.5] |
| | 64 | 16.0 | 16.2 | 15.8 | −0.2 [−1.7, +1.3] | +0.2 [−0.8, +1.2] |
| 1.7B B | 1 | 18.6 | 16.8 | 18.4 | +1.8 [+0.3, +3.2] | +0.2 [−0.9, +1.3] |
| | 4 | 17.2 | 15.4 | 16.8 | +1.8 [+0.3, +3.2] | +0.4 [−0.7, +1.6] |
| | 16 | 16.5 | 16.0 | 16.0 | +0.5 [−1.0, +2.0] | +0.5 [−0.8, +1.7] |
| | 64 | 15.1 | 14.7 | 15.3 | +0.4 [−1.2, +2.0] | −0.2 [−1.4, +1.1] |
| 4B A (closed 33.8) | 1 | 36.9 | 26.7 | 29.9 | +10.3 [+7.8, +12.7] | +7.0 [+5.1, +9.0] |
| | 4 | 17.1 | 14.8 | 16.4 | +2.2 [0.0, +4.5] | +0.7 [−1.1, +2.4] |
| | 16 | 5.1 | 4.8 | 3.3 | +0.3 [−1.3, +2.0] | +1.8 [+0.6, +2.9] |
| | 64 | 1.1 | 3.5 | 1.0 | −2.4 [−3.6, −1.4] | +0.1 [−0.3, +0.5] |
| 4B B | 1 | 36.7 | 32.5 | 34.2 | +4.2 [+2.7, +5.8] | +2.5 [+1.6, +3.6] |
| | 4 | 33.7 | 31.8 | 33.2 | +1.9 [+0.4, +3.4] | +0.5 [−0.5, +1.5] |
| | 16 | 31.9 | 32.0 | 32.4 | −0.1 [−1.7, +1.5] | −0.5 [−1.6, +0.6] |
| | 64 | 31.4 | 30.5 | 31.3 | +1.0 [−0.5, +2.4] | +0.1 [−1.2, +1.4] |

**Against the pre-registered criterion:** N\* (largest N where memory beats
both controls) is 0 for both 1.7B configs and 1 for both 4B configs, so
N\*(B) ≤ N\*(A) in both models: 32× more key space does **not** extend the
content signal to more passages. What it changes is the harm: at 4B,
N = 64, memory EM is 31.4 with D 4,096 vs 1.1 with D 128 (closed-book 33.8);
bigger keys cut interference noise, but what is left is not
answer-specific. Conclusion: the frozen Variant 1 read is limited by the
read itself (Qwen's query–key similarity does not pick out the right stored
tokens), not only by state size; at corpus scale capacity is an additional
limit that these runs do not measure.

Caveats: the gate was selected on a noisy criterion (calibration
differences ≤ 2.2 points). 1.7B A landed on the grid edge 0.05, a weak read
that also loses the earlier 1.7B N = 1 signal (+4.5 at gate 0.2), so the
1.7B rows are weak evidence (resolved by the equal-gate check below). 4B A used the same gate as the first 4B run
(0.2): its N = 1 signal reproduces (+10.3 vs +11.0), its N = 16 signal does
not. Outputs `runs/delta-rule/capacity/`, queue `run_capacity.sh`, gate
selection `select_capacity_gate.py`.

**1.7B sensitivity check at equal gates** (README amendment 2026-11-18;
labelled check, does not replace the pre-registered result). Because 1.7B A
was selected at the grid edge, A and B were compared at the same fixed
gates. Same split and CIs; closed-book 17.1:

| config, gate | N | memory | shuffled | cf | memory − shuffled | memory − cf |
|---|---:|---:|---:|---:|---|---|
| A, 0.1 | 1 | 19.8 | 16.6 | 17.5 | +3.2 [+1.7, +4.8] | +2.3 [+1.2, +3.5] |
| | 4 | 18.8 | 17.3 | 17.8 | +1.5 [−0.2, +3.1] | +1.0 [−0.4, +2.3] |
| | 16 | 15.5 | 15.8 | 14.6 | −0.3 [−2.1, +1.4] | +0.9 [−0.6, +2.2] |
| | 64 | 8.1 | 8.0 | 7.6 | +0.1 [−1.6, +1.8] | +0.5 [−0.6, +1.6] |
| B, 0.1 (= eval-B above) | 1–64 | 18.6 → 15.1 | | | see above | N\* = 0 |
| A, 0.2 | 1 | 21.0 | 15.6 | 17.2 | +5.4 [+3.3, +7.4] | +3.8 [+2.1, +5.6] |
| | 4 | 14.4 | 12.0 | 13.3 | +2.3 [+0.6, +4.2] | +1.1 [−0.7, +2.8] |
| | 16 | 2.1 | 1.8 | 2.1 | +0.4 [−0.7, +1.5] | +0.1 [−0.9, +1.1] |
| | 64 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B, 0.2 | 1 | 14.5 | 11.8 | 11.6 | +2.6 [+0.8, +4.5] | +2.8 [+1.4, +4.3] |
| | 4 | 13.7 | 13.1 | 12.2 | +0.6 [−1.2, +2.3] | +1.5 [+0.3, +2.7] |
| | 16 | 11.2 | 11.6 | 11.3 | −0.4 [−2.0, +1.3] | −0.1 [−1.5, +1.3] |
| | 64 | 5.8 | 6.3 | 6.0 | −0.6 [−1.9, +0.7] | −0.2 [−1.3, +0.9] |

N\*: gate 0.1 A 1, B 0; gate 0.2 A 1, B 1. So N\*(B) ≤ N\*(A) at both
gates: the 1.7B conclusion does not depend on the edge gate, and 1.7B now
matches 4B (signal at one passage only; bigger keys do not extend it but
make the read less harmful, e.g. N = 64 at gate 0.2: 5.8 vs 0.0). The
earlier caveat that the 1.7B rows are weak evidence is resolved by this
check. Queue `run_capacity_sens.sh`, outputs
`runs/delta-rule/capacity/1.7b/sens-*`.

### Global expanded-key rerun with both fixes (2026-11-18)

The largest expanded-key setting rerun on the full 4,342-passage memory with
both fixes (sink prefix, centred queries); same gate grids and `train_seen`
selection (README amendment 2026-11-18). These replace the D 4,096 rows of
the two expanded-key tables above as the reference; the other rows stay as
labelled (uncentred queries, first token dropped). EM %, 512 per group:

| model, rule | gate | train_seen | train_heldout | dev_in | dev_out | cf dev_in (follow) |
|---|---|---:|---:|---:|---:|---|
| 1.7B closed-book | — | 21.9 | 25.6 | 18.2 | 16.0 | — |
| 1.7B delta, D 4,096 | 0.1 | 9.0 | 10.7 | 8.4 | 9.0 | 7.2 (0) |
| 1.7B Hebbian, D 4,096 | 2 | 16.4 | 18.4 | 14.7 | 13.7 | 14.8 (0) |
| 4B closed-book | — | 39.7 | 40.6 | 35.2 | 32.4 | — |
| 4B delta, D 4,096 | 0.1 | 36.3 | 32.6 | 29.9 | 26.9 | 30.1 (0) |
| 4B Hebbian, D 4,096 | 2 | 37.1 | 37.1 | 31.8 | 28.7 | 31.6 (0) |

Paired dev_in CIs (memory − closed-book / memory − answer-swapped memory):

| model, rule | vs closed-book | vs answer-swapped |
|---|---|---|
| 1.7B delta | −9.8 [−12.9, −6.8] | +1.2 [+0.4, +2.1] |
| 1.7B Hebbian | −3.5 [−6.4, −0.8] | −0.2 [−0.6, 0.0] |
| 4B delta | −5.3 [−7.8, −2.7] | −0.2 [−1.0, +0.6] |
| 4B Hebbian | −3.3 [−6.1, −0.6] | +0.2 [−0.6, +1.0] |

**Against the criterion:** not met by any run; every run is below
closed-book. 1.7B delta is the only run where memory beats the swapped
memory (+1.2), a small content trace far smaller than the 9.8 points the
read costs. Compared with the flawed runs: 4B delta is less harmful (29.9
vs 25.0 dev_in), Hebbian is more harmful at both sizes (1.7B 14.7 vs 18.8,
4B 31.8 vs 34.0); centring the queries changes the read's scale, so the
same gate is a stronger read. All selected gates are again the smallest in
their grid, and larger gates give 0 EM; as before this is reported, not
re-tuned, because a smaller gate only moves toward `S = 0` = closed-book.
The answer does not change: with both fixes, expanded keys on the full
memory do not make the frozen heads read written facts. Outputs
`runs/delta-rule/expand-fixed/`, queue `run_expand_fixed.sh`, selection
`select_v1_expand.py {1.7b,4b} runs/delta-rule/expand-fixed/{1.7b,4b}`.

## Failure examples (dev_in, 1.7B, seed 0)

Questions that closed-book gets wrong and BM25@1 gets right, so the written
passage does contain the answer. The adapted memory (V1 LoRA Hebbian) answers
exactly what it answers with the answer-swapped memory, i.e. the passage is
never used:

| Question (shortened) | gold | swapped-in | closed-book | BM25@1 | V1 LoRA memory / swapped memory | V2 d 4096 |
|---|---|---|---|---|---|---|
| expression derived from a "ragman roll"? | rigmarole | johansberg | ragman roll | rigmarole | ragman roll / ragman roll | Ragman |
| term for a granite paving block (Belgian block)? | sett | johnson jack | cobbles | sett | cobbles / cobbles | Cobbles |
| substance that continually deforms under stress? | fluid | fabrics | Deformable substance | Fluid. | fluids / fluids | Newtonian |

Typical pattern: adaptation fixes answer *format* ("fluids" instead of a
paraphrase), which is why the LoRA rows gain the same with `S = 0`; no
condition ever outputs the swapped-in answer. Per-question predictions:
`runs/delta-rule/global-v1/runs/*/{memory,counterfactual,null}-*.jsonl`.

## Follow-up: small banks with training (2026-11-19 → 2026-11-25)

Requested by the maintainer's scepticism after the PR. Every step was
pre-registered as a dated README amendment before its results; full tables,
CIs and examples are in `NOTES.md` (sections of the same dates), analysis
scripts in `ci/`. Setup throughout: Qwen3-1.7B, banks-v2 eval (1,024
questions in banks of N consecutive questions; each question reads the
memory written from its bank's N passages), greedy exact match, paired
bootstrap 95% CIs (10,000 resamples, seed 20261111). Controls: *shuffled*
= N calibration passages; *counterfactual* = the bank's passages with the
answer swapped. N\* = largest N at which memory beats both with CI > 0.
Frozen Qwen closed-book: 17.1.

### Sanity checks (frozen Qwen, no training)

| check | finding |
|---|---|
| Store alone (`sanity_kv.py`) | Delta-rule store retrieves exact keys ~97–100% up to 2,048 facts (D 8,192); code is correct. |
| Answer injection (`sanity_inject.py`) | Memory → soft tokens → frozen Qwen: 21 → 62 EM at 256 facts with question keys; passage keys lose most of it. |
| Uncompressed softmax memory (`run_knn_upper.sh`) | Upper bound for frozen heads: 67.7 / 49.7 / 32.4 / 11.3 at N 1 / 4 / 16 / 64; N\* 64. |
| Whitened keys (`run_whiten.sh`) | Keys spread out (rank 18 → 88) but no content signal beyond N 1. |
| Self-read, QA-pair memory (`run_sanity3.sh`) | Storage of whitened + DPFP keys is nearly lossless to 64 passages; storing the question + answer gives +35 at N 1. |
| Cloze (`sanity_cloze.py`) | With identical query = key tokens the softmax memory copies the passage (98.6% at N 1) but the delta-rule memory never beats no-memory. |
| Variant 1T, trained feature maps (`run_trained.sh`) | Fails (N\* 0). |
| Read with the passage's mean query (`run_passage_query.sh`) | Knowing which passage to read does not help: ≤ +3 at N 1. |

Reading: storage works; the frozen linear read cannot pick the answer out of
a passage. Training the reading side is the remaining lever.

### Variant 1 + LoRA on small banks

Rank-8 LoRA on q_proj / o_proj (all layers) + per-head gates, trained on
banks of 1 / 4 / 16 TriviaQA *train* passages (no overlap with eval
questions or answers); memory written by plain Qwen (LoRA off). Control
LoRA: same recipe with unrelated memory (it learns to switch memory off).

| run | closed-book (same LoRA) | memory N = 1 / 4 / 16 / 64 | N\* |
|---|---|---|---|
| LoRA, β = 1 | 20.4 | 46.1 / 28.7 / 21.0 / 17.7 | 16 |
| + learned β per layer × head (option 1) | 17.5 | 51.4 / 34.3 / 22.7 / 15.9 | 16 |
| + learned per-token β (option 2) | 20.2 | **54.1 / 41.4 / 26.5** / 15.9 | 16 |
| option 2 + 4,096-wide keys (DPFP ν 16) | 19.8 | 53.2 / 41.2 / 27.0 / **18.9** | 64 |
| control LoRA (option 2 recipe) | 20.9 | 21.1 / 20.4 / 20.8 / 19.8 | 0 |

Option 2 at N 16: memory − shuffled +9.4 [+7.0, +11.8], − counterfactual
+9.5 [+7.2, +11.8], − own closed-book +6.2 [+3.8, +8.8]. The bigger memory
retrieves at N 64 (+3.2 [+1.1, +5.5] vs shuffled) but equals closed-book
there. Memory helps on questions frozen Qwen gets wrong (N 1, β = 1: 38.9%
vs 10.0% shuffled), so it is not just recall of known facts. Counterfactual
follow rate stays low (≤ 4%): the reader rejects the swapped answers, which
are usually of the wrong type for the question.

### Passages in context instead of memory (`run_context_check.sh`)

| N | in context, frozen | in context, option-2 LoRA | best memory |
|---|---|---|---|
| 1 | 70.9 | 67.3 | 54.1 |
| 4 | 68.4 | 67.9 | 41.4 |
| 16 | 62.3 | 67.9 | 27.0 |
| 64 | 53.8 | 64.7 | 18.9 |

The best memory reaches ~80% of the context score at N 1, ~40% at N 16
and nothing above closed-book at N 64.

### Variant 2 on small banks

**Step A** (`eval_v2_banks.py`, `run_v2_banks.sh`): every earlier V2/V2b
checkpoint, with or without the V1 LoRA, has N\* = 0 on small banks.

**Step B** (`run_v2q.sh`; `train_lora_banks.py --task cloze --writer-rank 8
--learn-beta token`): writer = Qwen's own k/v projections plus a rank-8
LoRA, applied only to what is written (Qwen's own pass is unchanged); reader
= q/o LoRA + gates + per-token β. Trained on WikiText fill-in-the-blank only
(no TriviaQA), as issue #6 requires for Variant 2.

| run | closed-book (same model) | memory N = 1 / 4 / 16 / 64 | shuffled | N\* |
|---|---|---|---|---|
| step B | 7.8 | **48.5 / 25.0 / 10.5** / 4.0 | 2.7 / 0.9 / 1.3 / 2.6 | 64 |
| step B + V1 LoRA stacked | 10.3 | 47.5 / 31.2 / 15.7 / 7.6 | 7.4 / 4.4 / 5.4 / 4.5 | 64 |
| control (unrelated memory) | 22.7 | 23.4 / 22.7 / 22.1 / 21.9 | 22.3 / 21.9 / 21.7 / 21.6 | 0 |

Pre-registered criteria: retrieves at every N; beats the control model at
N 1 (+25.1 [+21.4, +28.9]); beats its own closed-book at N 1 / 4 / 16.
Counterfactual follow 27.6% at N 1: it reads what was written. Caveat: the
WikiText-trained reader lowers closed-book recall (7.8 vs 17.1, one-word
cloze-style answers); against plain Qwen the memory is +31.4 / +7.9 at
N 1 / 4 and below it from N 16.

### Limits of the follow-up

One seed per configuration; one model size (1.7B); training banks of at
most 16 passages; small banks only (not the 4,342-passage corpus memory of
the original question).

## Provenance

- All runs were made from base commit `39f76cc` with the experiment code
  uncommitted (`working_tree_dirty: true` in every `meta.json`); the code in
  this PR is the code that produced them. Earlier code states are not
  preserved separately; code changes between runs are listed as amendments in
  `README.md`.
- `runs/` is gitignored, so `records/runs.jsonl` holds every run's
  `meta.json` (full arguments, seed, trainable parameters, optimizer steps,
  checkpoint step, data-manifest hash, prompt, summary metrics), every dataset
  `manifest.json` (source and output SHA-256s), training logs
  (`*.log.json`), `probe_v2.json` and the queue logs, one JSON line per file
  (`{"path", "content"}`; `pack_records.py`). Until 2026-11-26 these were
  ~730 separate files under `records/`; they are packed unchanged. Home
  directory paths in logs are replaced by `~`. Checkpoints (`interface.pt`, LoRA/gate weights), per-question
  predictions and the 16 GB hidden-state cache stay local under
  `runs/delta-rule/`.
- Test set and tuning: V1 gates/write parameters were tuned on calibration
  (train) questions, the expanded-key gate on `train_seen`; V2 on WikiText
  only. No dev question influenced any choice; the V2b pilots used WikiText
  validation only.

## Contamination: questions the model already knows

"Known" = answered correctly closed-book (dev_in 93/512, dev_out 82/512).
A memory that supplies facts should raise EM on *unknown* dev_in questions
only. EM % on known / unknown (seed-averaged):

| Condition | dev_in | dev_out |
|---|---|---|
| BM25@1 | 74.2 / **47.5** | 41.5 / 7.7 |
| V1 gates Hebbian | 92.8 / 3.9 | 80.9 / 5.3 |
| V1 LoRA Hebbian | 79.2 / 13.4 | 70.7 / 14.7 |
| V1 LoRA Hebbian, S = 0 | 78.5 / 13.2 | 75.6 / 15.3 |
| V1 LoRA trained with S = 0 | 78.9 / 13.0 | 72.8 / 15.2 |
| V2 d = 1024 | 71.0 / 6.7 | 64.6 / 7.9 |

Only RAG is dev_in-specific. Every memory condition moves dev_in and dev_out
alike, and the LoRA rows are indistinguishable from their empty-memory
controls.

## Exposure budget

| Variant | trainable params | memory floats | training data | steps × batch |
|---|---:|---:|---|---|
| V1 frozen | 0 | 3.7 M (28 × 8 × 128²) | — | — |
| V1 gates | 448 | 3.7 M | 3,318 TriviaQA-train questions × 2 | 416 × 16 |
| V1 LoRA | 1.84 M | 3.7 M | same | 416 × 16 |
| V1 expanded keys | 0 | 14.7 / 29.4 / 58.7 / 117 M | — | — |
| V2 d = 512 … 4096 | 4.2 / 8.4 / 16.8 / 33.6 M | 0.26 / 1.05 / 4.2 / 16.8 M | WikiText cloze, no TriviaQA | 2,000 × 32 |
| 4B V1 gates / LoRA | 1,152 / 3.84 M | 4.7 M (36 × 8 × 128²) | as 1.7B | 416 × 16 |
| 4B V1 expanded keys | 0 | 18.9 / 37.7 / 75.5 / 151 M | — | — |
| 4B V2 d = 512 … 4096 | 5.2 / 10.5 / 21.0 / 41.9 M | 0.26 / 1.05 / 4.2 / 16.8 M | WikiText cloze | 2,000 × 32 |
| 4B V2b d = 1024 | 10.5 M | 1.05 M | WikiText cloze | 1,000 × 32 |

Every condition writes the same 4,342 passages (183,667 tokens) once.

## What later stages may rely on

- Neither variant is a working factual memory at corpus scale here; do not
  build on them as such.
- Reusable pieces: exact chunk-parallel delta rule
  (`memwmlm.memory.delta_rule.write_chunked`), the global-memory protocol
  with dev_in/dev_out/counterfactual groups and its baselines, the
  right-vs-wrong-memory probe (`probe_v2.py`) as a cheap positive control
  for any trained memory interface, and the layer-hidden cache.
- Unproven: whether a trained interface can learn retrieval with a
  different objective (e.g. contrastive right-vs-wrong loss, retrieval
  pre-training), more steps, or reads at several layers; training V2b on
  large N for 4B; whether the 4B N = 1 small-bank signal can be made useful
  without the read's harm; whether a read with better query–key matching
  (trained or not) plus enough key space works at corpus scale.
- Follow-up (small banks): a trained reader (LoRA + per-token β) makes the
  delta-rule memory useful up to ~16 passages for Variant 1, and a
  general-text-trained writer/reader initialised from Qwen's heads retrieves
  for Variant 2. Open: scaling past 64 passages, the V2 reader's damage to
  closed-book recall, other seeds and model sizes.

## Reproduction

```bash
P=experiments/issue-6-delta-rule-memory
.venv/bin/python $P/prepare_banks.py --output runs/delta-rule/banks-v2
.venv/bin/python $P/prepare_banks.py --output runs/delta-rule/banks-first-v1 --selection first
.venv/bin/python $P/prepare_global.py
$P/run_v1_global.sh 0 frozen; for s in 0 1 2; do $P/run_v1_global.sh $s; done
.venv/bin/python $P/prepare_wikitext_cloze.py --output runs/delta-rule/wikitext-cloze-v1
.venv/bin/python $P/cache_hidden.py
$P/run_v2.sh 0 "512 1024 2048"; $P/run_queue_1116.sh   # d=4096 + expanded keys
$P/run_4b.sh   # Qwen3-4B; then, since V2b passed its rule:
.venv/bin/python $P/eval_v2.py --checkpoint runs/delta-rule/qwen3-4b/v2b/d1024-fewshot/interface.pt
$P/run_4b_sweeps.sh   # 4B V2 d 512/2048/4096 + expanded keys
$P/run_capacity.sh    # 2026-11-18 fixes + small-bank capacity test
.venv/bin/python $P/select_v1_expand.py 4b
$P/run_expand_fixed.sh   # 2026-11-18 global D 4,096 rerun with both fixes, then:
.venv/bin/python $P/select_v1_expand.py 1.7b runs/delta-rule/expand-fixed/1.7b
.venv/bin/python $P/select_v1_expand.py 4b runs/delta-rule/expand-fixed/4b
BOOTSTRAP_REF=runs/delta-rule/qwen3-4b/global/frozen-delta .venv/bin/python $P/bootstrap_global.py <4B run dirs>
.venv/bin/python $P/probe_v2.py --checkpoint runs/delta-rule/v2/d2048-s0/interface.pt
.venv/bin/python $P/analyze_global.py; .venv/bin/python $P/bootstrap_global.py <run dirs>
.venv/bin/python $P/select_v1_expand.py
```

Follow-up (2026-11-19 → 25), in order:

```bash
.venv/bin/python $P/sanity_kv.py; .venv/bin/python $P/sanity_inject.py
$P/run_knn_upper.sh; $P/run_whiten.sh; $P/run_sanity3.sh
.venv/bin/python $P/train_feature_map.py; $P/run_trained.sh   # V1T + cloze check
$P/run_passage_query.sh; $P/run_passage_query_fixed.sh
$P/run_lora_banks.sh; $P/run_lora_beta.sh; $P/run_lora_tbeta.sh; $P/run_lora_dtbeta.sh
$P/run_context_check.sh; $P/run_v2_banks.sh; $P/run_v2q.sh
.venv/bin/python $P/ci/<script>.py   # tables in NOTES.md (see ci/README.md)
.venv/bin/python $P/pack_records.py  # refresh records/runs.jsonl
```

Every run directory has a `meta.json` with all arguments, data manifest
hashes, repository commit and `working_tree_dirty`. Runs so far were made on
an uncommitted working tree on branch `delta-rule` (base `39f76cc`).
