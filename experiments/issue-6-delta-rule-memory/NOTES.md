# Issue #6 working log (local, not yet posted)

Chronological log of what was run and observed. Final numbers go to
`RESULTS.md`; this file keeps diagnostics and decisions with their provenance.

Environment: `.venv` (Python 3.13, torch 2.15.0+cu132, transformers 5.22.2),
RTX 5090 Laptop 24 GB, Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
bf16. All runs below are from an uncommitted working tree on branch
`delta-rule` (base `39f76cc`); commit SHAs will be attached before reporting.

## 2026-11-11 — Data

- TriviaQA RC archive SHA-256
  `ef94fac6db0541e5bb5b27020d067a8b13b1c1ffc52717e836832e02aaed87b9`
  (2,665,779,500 bytes). `qa/wikipedia-train.json`
  `05752644…ab635`, `qa/wikipedia-dev.json` `50657773…effdf`.
- Banks: `runs/delta-rule/banks-v2` (rule
  `max_question_overlap_answer_sentence_or_window_v2+cf_all_aliases_v2`, seed
  20261111): 1,024 eval (dev) + 1,024 calibration (train) questions; median
  passage 27 words, max 64; all 1,024 counterfactuals valid (no original alias
  left, donor answer not pre-present).
- 413 over-drawn dev questions were dropped because an alias also answers a
  calibration question. This biases the eval set away from very common answers.
- Passage selection maximises question-word overlap, which **favours lexical
  retrieval (BM25)**. A question-independent selection bank is planned as a
  robustness check before any memory-vs-RAG claim.

## 2026-11-11 — Plumbing smoke (16 eval questions; not used for any choice)

- `null_memory` reproduces `closed_book` exactly (as do unit tests on a tiny
  Qwen3), confirming the read path is inert at zero state.
- Variant 1 with gate 1.0 on all 28 layers destroys generation (outputs of
  dots / empty strings): frozen injection at unit gate is off-distribution.

## 2026-11-11 — Diagnostic 1: do frozen heads address the right passage?

`probe_head_retrieval.py`, calibration split, 256 questions. Score = max cosine
between any pre-RoPE query token of the closed-book prompt and any key of a
passage; top-1 = own passage ranked first among the N in its bank.

| N | chance | BM25 | best head (maxsim) | best head (last token) | median head (maxsim) |
|---|---:|---:|---:|---:|---:|
| 16 | 0.063 | 0.902 | 0.723 (L11 H3) | 0.520 (L22 H14) | 0.082 |
| 64 | 0.016 | 0.902 | 0.477 (L16 H14) | 0.289 (L22 H14) | 0.023 |
| 256 | 0.004 | 0.848 | 0.297 (L16 H14) | 0.094 (L20 H4) | 0.004 |

Reading: a handful of frozen mid-layer heads carry real content addressing
(tens of × chance); the median head is at chance. A uniform read over all heads
mixes a few informative heads with many noise heads. BM25 is inflated by the
selection rule (above).

## 2026-11-11 — Diagnostic 2: what does a 128×128 head state retain?

`probe_state_fidelity.py`, calibration split. Write every token of N passages
(first token skipped) into per-layer/KV-head states; read with (a) the written
keys, (b) keys of 16 passages that were **never written**. Metric: cosine
between read-out and true value after removing the mean value (specific
content). Values are means over all 28×8 heads.

| keys | N (tokens) | delta: written / unwritten | best-layer gap | Hebbian: written / unwritten |
|---|---|---|---:|---|
| raw | 1 (36) | 0.29 / 0.15 | 0.31 | −0.01 / −0.25 |
| raw | 16 (594) | 0.31 / 0.25 | 0.09 | −0.01 / −0.06 |
| raw | 256 (10.6k) | 0.45 / 0.44 | 0.02 | −0.01 / −0.02 |
| centred | 1 (36) | 0.61 / 0.26 | 0.55 | 0.63 / 0.23 |
| centred | 4 (128) | 0.50 / 0.28 | 0.36 | 0.51 / 0.25 |
| centred | 16 (594) | 0.43 / 0.34 | 0.14 | 0.41 / 0.32 |
| centred | 64 (2.6k) | 0.40 / 0.38 | 0.04 | 0.37 / 0.34 |
| centred | 256 (10.6k) | 0.40 / 0.40 | 0.01 | 0.36 / 0.35 |

Readings:

1. Pre-RoPE keys are highly collinear (mean |cos| between distinct keys
   0.74–0.78); with raw keys the Hebbian state stores nothing specific. Centring
   keys (subtract the bank mean; calibration, no training) reduces this to ~0.22.
2. **Written ≈ unwritten for N ≥ 64**: at that point the state is a generic
   linear k→v regression (k and v are both linear in the same hidden state), not
   a memory of the passages. Self-retrieval fidelity alone is therefore a
   misleading metric; the unwritten-key control is required.
3. Writing *every token* of the passages saturates a 128-dim head state within a
   few hundred tokens (N ≈ 4–16 passages). Global Variant 1 memory over a
   realistic bank cannot work in this form.

Design consequences (to be recorded as amendments before eval-memory runs):

- V1 needs **sparse writes** (a few tokens per passage, e.g. the answer-bearing
  or last-token positions) and/or **head selection** (read only heads that
  address, chosen on calibration data), and key centring.
- The `d` sweep of Variant 2 is the direct capacity test; Variant 1 is fixed at
  128 per head (28 × 8 states).

## 2026-11-11 — V1-frozen calibration sweep 1 (`sweep_v1_frozen.sh`)

Calibration questions 256–383 (128), key centring on, delta rule. Closed-book
25.8% (= null memory at N=1; 26.6% null at N=16).

- Probe-selected heads (top 8 / 32 by maxsim at N=16), gate 0.5–4: memory never
  above shuffled or closed-book; gate ≥ 2 collapses output.
- All heads, gate 0.05 / 0.1: ≈ closed-book (inert). Gate 0.25: **N=1 memory
  34.4% vs shuffled 18.8% vs closed-book 25.8%**; N=16 collapses to 0%.
- Paired vs closed-book at N=1, gate 0.25: 16 memory-only vs 5 closed-only
  correct. Most memory-only answers are strings present in the written passage
  that closed-book got wrong (Haiti/Cuba, Edward III/VI, Horrible
  Histories/Magic School Bus, Alaska/Maine, Jerusalem/Acre, The Hobbit/The
  Silmarillion).

Reading (provisional, 128 questions): frozen heads + delta-rule state transfer
passage content into answers for a single written passage, in a narrow gate
window; nothing survives at N=16, consistent with diagnostic 2. Probe head
selection did not help — addressing (maxsim) is not the same as linear readout.

Next: refine gate × N ∈ {1,2,4} on fresh calibration questions 384–895, then
fix one configuration *before* running eval memory conditions.

## 2026-11-11 — Eval baselines (banks-v2, 1,024 dev questions)

Output: `runs/delta-rule/eval-baselines/`. Exact match (%):

| Condition | N=1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| closed_book | 17.09 | | | | | |
| oracle_passage | 70.90 | | | | | |
| full_context | (70.90) | 68.36 | 62.30 | 53.81 | 43.07 | n/a (>30k tokens) |
| rag_bm25@1 | 70.90 | 67.48 | 66.89 | 65.14 | 62.40 | 59.08 |
| rag_bm25@3 | 70.90 | 67.87 | 66.21 | 64.16 | 61.33 | 57.32 |

- Closed-book 17.09% agrees with issue #5 (16.72% on unfiltered-web dev).
- `counterfactual_oracle`: follows the swapped answer 32.2%, original 7.4%. The
  swap is not type-matched, so ~32% is the in-context ceiling for the
  counterfactual-follow metric.
- Full context degrades faster with N than BM25; BM25 is favoured by the
  overlap-based selection rule (robustness bank `banks-first-v1` pending).

## 2026-11-11 — Robustness bank: question-blind passage selection

`banks-first-v1` (`--selection first`: first answer-bearing sentence in
document order; same 1,024 eval question ids). Output
`runs/delta-rule/eval-baselines-first/`. EM (%):

| Bank | oracle | BM25@1 N=4 | 16 | 64 | 256 | 1024 | full N=4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overlap (v2) | 70.9 | 67.5 | 66.9 | 65.1 | 62.4 | 59.1 | 68.4 | 62.3 | 53.8 |
| question-blind | 67.9 | 54.9 | 51.3 | 47.7 | 42.5 | 37.6 | 65.5 | 57.2 | 45.1 |

The overlap selection rule inflated BM25 by 13–22 points at N ≥ 4 while the
passages themselves are only ~3 points more useful (oracle). Memory-vs-RAG
comparisons must use the question-blind bank or report both. Counterfactual
in-context follow rate on this bank: 27.1%.

## 2026-11-11 — V1-frozen on eval (fixed config from the amendment)

Output `runs/delta-rule/eval-v1-frozen/` (summary `summary.json`), banks-v2,
1,024 dev questions, delta rule, β=α=1, all heads, gate 0.2, centred keys.
EM (%), paired Δ vs closed-book 17.09 with bootstrap 95% CI:

| Condition | N=1 | N=2 | N=4 | N=16 |
|---|---:|---:|---:|---:|
| memory (delta) | **21.6** (+4.5 [+2.4,+6.5]) | 18.8 (+1.8 [−0.3,+3.8]) | 13.3 (−3.8) | 0.4 |
| null_memory | 17.1 | 16.7 | 16.8 | 17.1 |
| shuffled_memory | 15.7 (−1.4) | 13.4 (−3.7) | 10.5 (−6.5) | 0.9 |
| counterfactual_memory | 17.4 (+0.3) | 14.6 | 11.0 | 0.4 |
| memory (Hebbian, untuned) | 10.3 | 0.4 | 0.0 | 0.0 |

Paired at N=1: memory − shuffled +5.9 [+3.8,+7.9]; memory − counterfactual
+4.2 [+2.4,+6.0]. EM on closed-book-wrong questions: memory 9.4% vs shuffled
4.5% (oracle passage in context: 68.0%). Closed-book→memory flips: 80 gained,
34 lost; 67/80 gained predictions are substrings of the written passage.

Counterfactual follow rate: **0.0%** at every N (also with a loose
donor-alias-substring check), vs 32.2% for the same swap in context. Of the 80
gained answers, only 26 survive the swap: replacing the answer span removes
most of the gain, but the donor answer is never produced.

Reading: Variant 1 frozen memory carries real, answer-span-dependent
information for a single passage (beats shuffled and counterfactual by 4–6
points, CIs exclude 0) but only ~6% of the oracle-in-context gain, nothing
above closed-book from N=4, and harm at N=16. It does not make the model state
a new (counterfactual) fact; it acts as a weak nudge toward a candidate the
model can already produce. By the protocol's falsification clause ("gains do
not follow counterfactual swaps") **C1 is refuted for V1-frozen**; the
answer-span-dependent component is a weaker, reportable positive. Hebbian is
worse than null everywhere (ablation, untuned).

Caveat (2026-11-12): the Hebbian row above used β = 1 with no normalisation,
so its state grows with the number of tokens written (read magnitude ∝ T);
the collapse at N ≥ 2 is at least partly a scale artefact, not evidence about
Hebbian storage. A normalised rerun is part of the write-parameter sweep
(README amendment 2026-11-12).

## 2026-11-12 — Write-parameter sweep, Stages A and C (calibration 384–895)

`runs/delta-rule/v1-write-sweep/`, scored by `score_sweep.py`. Closed-book on
this slice 20.51%. Selection score = mean over N ∈ {1,2,4,16} of
(memory − shuffled) EM. Selected rows (memory / shuffled EM at N = 1, 2, 4, 16):

| config | score | memory | shuffled |
|---|---:|---|---|
| delta β=0.5 g=0.3 (**selected delta**) | 6.05 | 27.1 / 21.5 / 15.0 / 0.2 | 17.0 / 13.9 / 8.8 / 0.0 |
| Hebbian β=1/T g=4 (**selected Hebbian**) | 5.81 | 25.6 / 23.6 / 21.3 / 18.9 | 14.1 / 16.2 / 18.6 / 17.4 |
| delta β=0.3 g=0.5 | 5.52 | 28.1 / 17.2 / 4.9 / 0.0 | 15.8 / 9.8 / 2.5 / 0.0 |
| delta β=1 g=0.2 (previous eval config) | 4.30 | 26.8 / 20.7 / 13.7 / 1.0 | 17.6 / 15.4 / 11.3 / 0.6 |
| delta β=0.5 g=0.2 | 4.25 | 25.0 / 24.4 / 19.5 / 8.6 | 20.9 / 17.4 / 16.2 / 6.1 |
| Hebbian β=1/T g=2 | 3.27 | 26.8 / 23.4 / 22.1 / 20.1 | 19.3 / 19.9 / 19.7 / 20.3 |

Readings:

1. β < 1 helps the delta rule moderately (β=0.5 beats β=1 at the same gate at
   every N) but every delta setting still collapses at N = 16 unless the gate
   is so small that memory is inert.
2. Token-normalised Hebbian (state = mean of kvᵀ) **degrades gracefully**: at
   g=2 it is ≥ closed-book at all N (26.8 → 20.1) instead of collapsing — the
   read becomes an average over all written tokens, so the own-passage signal
   dilutes with N rather than corrupting generation. The earlier Hebbian
   collapse was a scale artefact, as suspected.
3. **Weakness of the pre-registered criterion:** memory − shuffled rewards
   configurations in which shuffled memory is *more harmful* than real
   memory, even if real memory is below closed-book (selected delta config at
   N = 4: 15.0 vs 20.5 closed-book). The criterion is kept for the selected
   eval runs as pre-registered; this weakness is recorded here and will be
   stated in the results. Any additional "no-harm" configuration run on eval
   (e.g. Hebbian g=2) must be labelled post-hoc.

Stage B (α ∈ {0.999, 0.99, 0.95} at β = 0.5, gates 0.2/0.3/0.5) started.

## 2026-11-12 — Write-parameter sweep, Stage B (α at β = 0.5)

| config | score | memory N=1/2/4/16 | shuffled N=1/2/4/16 |
|---|---:|---|---|
| delta β=0.5 α=0.99 g=0.5 (**selected delta, final**) | 7.86 | 27.5 / 19.9 / 13.5 / 10.5 | 14.1 / 11.3 / 8.2 / 6.4 |
| delta β=0.5 α=1 g=0.3 (Stage-A winner) | 6.05 | 27.1 / 21.5 / 15.0 / 0.2 | 17.0 / 13.9 / 8.8 / 0.0 |
| delta β=0.5 α=0.95 g=0.5 | 4.59 | 26.6 / 23.0 / 20.3 / 19.3 | 17.2 / 18.9 / 17.2 / 17.6 |

α < 1 removes the N = 16 collapse, but for a mundane reason: decay is per
written token (α^36 ≈ 0.70 per passage at α = 0.99), so the state is dominated
by the last few passages written and the rest are forgotten. That is a recency
buffer, not a global memory; the eval will check accuracy by the passage's
write position within its bank. Same criterion weakness as above: the
selected config is below closed-book (20.5) at N = 2, 4, 16.

Eval runs (banks-v2, all memory controls, N ∈ {1,2,4,16}):
pre-registered selections `d-b0.5-a0.99-g0.5` and `h-bT-a1-g4`; post-hoc
(labelled as such) `h-bT-a1-g2`, the best no-harm configuration.

## 2026-11-12 — Tuned V1-frozen on eval (`runs/delta-rule/eval-v1-tuned/`)

banks-v2, 1,024 dev questions, closed-book 17.09, oracle 70.90. EM (%) and
paired differences with bootstrap 95% CI:

| config | N | memory | Δ closed-book | memory − shuffled | memory − counterfactual | cf follow |
|---|---:|---:|---|---|---|---:|
| delta β=0.5 α=0.99 g=0.5 (pre-reg) | 1 | 25.1 | +8.0 [+5.6,+10.5] | +12.7 [+10.3,+15.1] | +11.2 [+9.1,+13.5] | 0.1 |
| | 2 | 15.5 | −1.6 | +6.0 [+3.8,+8.2] | +5.4 [+3.5,+7.2] | 0.0 |
| | 4 | 10.3 | −6.8 | +2.1 [+0.0,+4.1] | +2.8 [+1.4,+4.3] | 0.0 |
| | 16 | 6.0 | −11.1 | −0.7 | +0.0 | 0.0 |
| Hebbian β=1/T g=4 (pre-reg) | 1 | 23.4 | +6.3 [+3.9,+8.9] | +11.3 [+8.9,+13.9] | +11.1 [+9.2,+13.3] | 0.0 |
| | 2 | 19.1 | +2.1 [+0.2,+3.9] | +4.6 [+2.7,+6.5] | +2.8 [+1.5,+4.3] | 0.0 |
| | 4 | 17.3 | +0.2 | +1.4 [−0.3,+2.9] | +1.6 [+0.4,+2.8] | 0.0 |
| | 16 | 17.0 | −0.1 | +0.6 | +0.5 | 0.0 |
| Hebbian β=1/T g=2 (post-hoc) | 1 | 22.3 | +5.2 [+3.4,+7.0] | +5.0 [+3.1,+6.8] | +5.3 [+3.8,+6.8] | 0.0 |
| | 2 | 19.1 | +2.1 [+0.7,+3.4] | +2.0 | +1.5 | 0.0 |
| | 4 | 18.2 | +1.1 [−0.2,+2.4] | +0.9 | +0.3 | 0.0 |
| | 16 | 17.2 | +0.1 | +0.4 | +0.0 | 0.0 |

(Previous pre-registered config, delta β=α=1 g=0.2: N=1 21.6, +4.5.)

Write position at N = 16 (memory EM, passage written in slots 0–7 / 8–15 /
14–15): delta α=0.99 5.9 / 6.1 / 7.8; Hebbian g=4 15.8 / 18.2 / 19.5;
Hebbian g=2 15.6 / 18.8 / 21.1. Weak recency only; the α=0.99 "recency
buffer" did not transfer from calibration (10.5 there, 6.0 here).

Readings:

1. Tuning the write parameters roughly doubles the single-passage gain
   (+4.5 → +8.0 EM, i.e. ~15% of the oracle-in-context gain of +53.8) and
   makes the gain clearly passage-specific (memory beats shuffled and
   counterfactual memory by 11–13 points at N = 1).
2. Nothing survives past N ≈ 2–4 for any rule or setting: delta harms
   generation, normalised Hebbian dilutes to ≈ closed-book.
3. Counterfactual follow stays at **0%** in every configuration (in-context
   reference 32%). Frozen reads never make the model output a fact it does
   not already consider plausible.
4. Conclusion for V1-frozen is unchanged and now robust to α, β, gate and
   rule: C1 refuted; a small, passage-specific, answer-span-dependent
   priming effect exists only for 1–2 passages.

## 2026-11-13 — Global memory, Variant 1, seed 0 (`runs/delta-rule/global-v1/runs/`)

One S per layer/KV head over the whole corpus (4,342 passages, ~160k tokens;
README amendment 2026-11-13). EM (%), 512 questions per group; "null" = same
trained parameters with S = 0.

| run | train_seen mem/null | train_heldout mem/null | **dev_in** mem/null | dev_out mem/null | dev_in cf (follow) |
|---|---|---|---|---|---|
| closed-book | 21.9 | 25.6 | 18.2 | 16.0 | |
| BM25@1 over corpus | 55.1 | 55.5 | 52.3 | 13.1 | |
| frozen delta g=0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (0) |
| frozen Hebbian g=2 | 22.7 | 25.8 | 17.2 | 16.6 | 17.4 (0) |
| frozen Hebbian g=4 | 19.1 | 23.2 | 16.8 | 16.8 | 16.6 (0) |
| gate delta | 0.0 / 21.9 | 0.0 / 25.6 | 0.0 / 18.2 | 0.0 / 16.0 | 0.0 (0) |
| gate Hebbian | 24.0 / 21.9 | 28.5 / 25.6 | 19.9 / 18.2 | 17.4 / 16.0 | 19.9 (0) |
| LoRA delta | 12.3 / 25.0 | 13.5 / 28.1 | 9.2 / 20.5 | 9.0 / 19.5 | 9.2 (0) |
| LoRA Hebbian | 33.6 / 33.2 | 32.6 / 34.0 | 24.6 / 24.8 | 22.7 / 23.8 | 24.8 (0) |
| LoRA trained with S=0 (control) | – / 33.4 | – / 32.2 | – / 24.4 | – / 24.2 | |

Paired (bootstrap 95% CI), gate Hebbian, memory − null: train_heldout +2.9
[+0.8,+5.1], dev_in +1.8 [+0.0,+3.5], dev_out +1.4 [−1.0,+3.7]; memory −
counterfactual on dev_in **+0.0 [0,0]** (all 512 predictions identical when
the 512 dev_in passages are answer-swapped). LoRA Hebbian memory − null on
dev_in −0.2 [−1.6,+1.0]; vs the S=0-trained LoRA control +0.2 [−1.2,+1.6].

Readings:

1. **No fact-specific memory effect at global scale in any stage.** The only
   positive difference (gate Hebbian, ≈ +2) is the same whether or not the
   question's passage is in S (dev_in vs dev_out) and is unchanged when the
   stored answers are swapped: S acts as a fixed bias vector, not a memory.
2. LoRA on TriviaQA-train lifts EM by ~6 points on dev **with or without
   memory** (control 24.4 vs closed-book 18.2): plain fine-tuning gain, and
   reading S adds nothing on top of it.
3. Global delta state is ill-conditioned: mean Frobenius norm per head 1,351
   (Hebbian 5.5). Writing ~160k tokens of correlated keys makes the delta
   rule a near-singular least-squares fit; reading it with *queries* gives
   huge outputs and destroys generation (frozen 0%, trained gates do not
   recover it within 416 steps, LoRA only partly: 9%).
4. Seen vs novel: no memorisation signal either — `train_seen` gains are the
   same size as `train_heldout`.

Seeds 1 and 2 running.

## 2026-11-13 — Global Variant 1, seeds 0/1/2

dev_in EM memory / null (seeds 0/1/2): gate Hebbian 19.9/20.1/20.1 vs 18.2;
LoRA Hebbian 24.6/25.6/25.8 vs null 24.8/24.6/25.8 and S=0-trained control
24.4/24.6/26.0; LoRA delta 9.2/10.6/9.0; gate delta 0 in all seeds.
Counterfactual EM equals memory EM to within 0.4 in every seed; follow 0%.
Seed-0 conclusions hold across seeds. (The `memory-*` rows of the lora-null
control are 0%: reading a delta S through a reader trained with S = 0 is
undefined behaviour, not a result.)

## 2026-11-13 — Variant 2 seed 0 (pre-registered protocol): no retrieval learned

Training (2,000 steps, WikiText cloze): loss 5.0 → 2.91/2.90/2.88 for
d = 512/1024/2048; val EM (N = 1,024) 0.8 → 3.5/3.9/4.7. d = 4096 OOM at
step ~575 (2.57 GiB allocation on a long N = 2,048 episode); not rerun.

eval_v2 (global corpus 4,342 passages), EM %:

| d | train_seen | train_heldout | dev_in | dev_out | cf dev_in (follow) | dev_in @512/1024/2048 | cloze N1024 / N4342 / closed |
|---|---|---|---|---|---|---|---|
| 512 | 18.8 | 22.5 | 17.8 | 15.4 | 17.6 (0) | 17.0/17.2/17.4 | 2.7 / 2.0 / 1.4 |
| 1024 | 20.1 | 23.2 | 18.4 | 17.0 | 18.8 (0) | 17.8/18.4/18.0 | 2.7 / 2.3 / 1.4 |
| 2048 | 20.3 | 22.3 | 17.6 | 15.8 | 18.0 (0) | 17.4/17.2/17.4 | 2.9 / 2.9 / 1.4 |

Closed-book references: 21.9 / 25.6 / 18.2 / 16.0. Memory is at or slightly
below closed-book everywhere; counterfactual changes nothing.

`probe_v2.py` (d = 2048, WikiText val, 256 queries): teacher-forced loss
with the real S vs with S of N *other* sentences vs S = 0:

| N | real | wrong | null |
|---|---|---|---|
| 1 | 7.06 | 7.04 | 7.75 |
| 8 | 4.47 | 4.45 | 7.35 |
| 64 | 3.16 | 3.15 | 7.49 |
| 512 | 2.96 | 2.97 | 6.95 |
| 2048 | 2.91 | 2.91 | 7.05 |

**Real = wrong at every N, including N = 1.** The whole loss drop is a
content-independent read (the interface learned the cloze answer format as a
generic bias); there is no retrieval at all. So the pre-registered V2 is a
null result for an uninformative reason: the positive control failed.
Likely cause: key and value come from the same token, so a context query can
only retrieve context, never the token that follows it; and the generic-bias
shortcut supplies all the early gradient. See amendment V2b.

## 2026-11-13 — V2b development pilots (WikiText val only, d = 1024)

Each pilot trains a fresh interface; "gap" = teacher-forced loss with the
real S vs S of other sentences (32 queries, noisy).

| pilot | changes | steps | end gap N16 | end gap N64 | val EM (N1024) / closed |
|---|---|---|---|---|---|
| shift | next-token values, N ∈ {32..2048} | 500 | — | 3.221 / 3.220 | 1.2 / 0.8 |
| +S0 small | + learned constant state S0, N ∈ {1,4,16} | 500 | 3.542 / 3.537 | 3.269 / 3.259 | 0.8 / 0.8 |
| +std | + fixed per-dim standardisation of h | 500 | 3.577 / 3.566 | 3.327 / 3.331 | 0.0 / 0.8 |
| sanity | + full-sentence query (only the span blanked), lr 1e-3 | 1000 | 3.023 / 2.947 | 2.788 / 2.911 | 5.9 / 2.3 |

Diagnostics on the way:
- Gradients reach every interface parameter; W_K/W_V/W_Q norms move only
  ~4% from init at lr 3e-4 (W_O, S0 absorb the training signal).
- Layer-14 hidden states have a massive-activation dim (1999: mean |x| 25 vs
  median 0.44). With RMS input, keys have mean pairwise cosine 0.46 at
  init and 0.65 after V2 training; with per-dim standardisation 0.01.

`probe_v2.py` on the sanity pilot (256 queries; null = no read at all):

| N | real loss | wrong loss | null loss | EM real / wrong / null |
|---|---|---|---|---|
| 1 | 3.034 | 3.108 | 6.833 | 5.9 / 5.5 / 3.5 |
| 4 | 2.655 | 2.701 | 6.013 | 8.2 / 7.8 / 5.5 |
| 16 | 2.999 | 3.030 | 6.195 | 5.5 / 3.9 / 3.1 |
| 64 | 2.913 | 2.971 | 6.180 | 8.2 / 7.0 / 3.9 |
| 1024 | 3.006 | 2.964 | 6.679 | 1.6 / 1.6 / 2.3 |

First consistent content effect (real < wrong at N ≤ 64), but tiny: 0.03–0.07
nats, while the format effect (wrong vs null) is ~3.5 nats. With the whole
sentence in the query and one stored sentence, a working memory should make
the answer nearly certain; this one does not. Nothing at N = 1,024.
Reading: with this architecture and budget, gradient descent finds the
content-independent format adaptation and almost no retrieval.

## 2026-11-16 — V2b few-shot pilot: decision rule fails → Variant 2 negative

`--fewshot 3` (chat turns) `--standardise --shift-values`, cropped cloze,
N ∈ {1,4,16}, lr 1e-3, 1,000 steps, d = 1024. Note first: with 3 shots the
*untrained* no-memory loss only drops 6.72 → 6.47, so the cropped cloze is
mostly hard (obscure names), not mis-formatted; trained interfaces still
reach ~3.3 with a *wrong* S, i.e. they learn a content-independent prior of
the WikiText answer distribution.

`probe_v2.py` (256 queries):

| N | real loss | wrong loss | null loss | EM real / wrong / null |
|---|---|---|---|---|
| 1 | 3.385 | 3.613 | 7.033 | 1.6 / 1.2 / 1.6 |
| 4 | 3.134 | 3.222 | 6.720 | 3.1 / 2.0 / 3.1 |
| 16 | 3.329 | 3.362 | 6.982 | 2.3 / 2.0 / 1.6 |
| 64 | 3.404 | 3.388 | 6.893 | 1.6 / 1.6 / 2.0 |
| 1024 | 3.316 | 3.258 | 6.734 | 0.8 / 1.6 / 1.6 |

Pre-registered rule (README 2026-11-16): N = 1 loss gap 0.228 ≥ 0.1 ✓;
N = 64 real EM > wrong EM ✗ (1.56 = 1.56). **Fails → Variant 2 reported
negative.** The trained interface stores a little content for a handful of
sentences (largest real-vs-wrong effect in all pilots), but nothing that
survives N ≥ 64, far below the 4,342-passage corpus. Per the rule, the
pre-registered V2 is completed at d = 4096 (write chunk 512).

## 2026-11-16 — V2 d = 4096 and V1 expanded keys (`run_queue_1116.sh`)

V2 pre-registered, d = 4096, seed 0, write chunk 512 (`runs/delta-rule/v2/d4096-s0/`):
train_seen 18.55, train_heldout 23.44, dev_in 15.62, dev_out 16.02, cf dev_in
15.62 (follow 0), dev_in at corpus 512/1024/2048 15.23 each, WikiText cloze
N 1,024 / 4,342 / closed 2.54 / 2.54 / 1.37. dev_in vs closed-book −2.5
[−5.3, +0.2], vs counterfactual 0.0 [−1.2, +1.0]. Same picture as d ≤ 2048.

V1 expanded keys (`runs/delta-rule/global-v1/expand/`, gate by train_seen,
`select_v1_expand.py`): table in RESULTS. Hebbian g2 at every D ≈ closed-book
(dev_in 18.6–19.3), g8 already 3–5%, g ≥ 32 0%. Delta: only g 0.1 non-zero
(4–11%), all larger gates 0%; selected gate sits on the grid edge, EM falls
monotonically with the gate, so extending the grid downwards can only approach
closed-book (S → 0). cf follow 0 everywhere; dev_in ≈ dev_out; memory ≈
counterfactual (CIs in RESULTS). Capacity is not the bottleneck.

## 2026-11-16 — Correction: 1.7B small-bank signal persists to N = 4

Running the same paired test as for 4B on `runs/delta-rule/eval-v1-frozen`
(pre-registered delta g 0.2; bootstrap 10,000, seed 20261111), memory minus
shuffled / counterfactual memory: N = 1 +5.9 [+3.8, +8.0] / +4.2 [+2.4, +6.0];
N = 2 +5.5 [+3.5, +7.4] / +4.3 [+2.6, +6.0]; N = 4 +2.7 [+1.0, +4.5] / +2.2
[+0.5, +4.0]; N = 16 −0.5 [−1.2, +0.1] / 0.0 [−0.4, +0.4]. The earlier
statements "only for 1–2 passages" / "nothing survives past N ≈ 2–4" (entry
above, and the first RESULTS draft) understated it: a small answer-dependent
difference is still there at N = 4 and gone at N = 16. Absolute EM falls
regardless (21.6 → 13.3 → 0.4 vs closed-book 17.1).

## 2026-11-16 — Qwen3-4B check (`runs/delta-rule/qwen3-4b/`, `run_4b.sh`)

Restarts: first adapted run OOM → `--micro-batch 8` (README note); V2b OOM
at step 1 → `train_v2.py --micro-batch 8` (README note). Steps 1–5 otherwise
as planned.

References (512 per group): closed-book train_seen 39.65, train_heldout
40.62, dev_in 35.16, dev_out 32.42; BM25@1 64.26 / 65.23 / 60.55 / 29.30.
dev_in BM25@1 vs closed-book +25.4 [+20.5, +30.3].

Frozen global V1 (gate by train_seen): delta g 0.1 (train_seen 0.78; g 0.3,
1 → 0) dev_in 0.39, dev_out 0.59, cf 0.20 (follow 0). Hebbian g 2
(train_seen 38.48 / 39.45 / 38.87 / 24.22 at g 1/2/4/8) dev_in 34.57, dev_out
32.23, train_heldout 40.23, cf 34.57 (follow 0).

Adapted global V1 (dev_in memory / S=0 / cf, seeds 0/1/2): gate delta
28.3/19.0/18.8, 35.2, 28.7/19.7/19.5; gate Hebbian 35.2/35.4/35.4, 35.2,
35.0/35.4/35.2; LoRA delta 25.6/23.4/32.4, 35.5/35.2/36.1, 25.6/23.0/31.8;
LoRA Hebbian 36.9/39.1/36.5, 37.5/40.4/37.5, 36.7/39.3/36.7; LoRA trained
S=0: memory 0 (all seeds), S=0 37.3/38.9/37.7. cf follow 0 everywhere. Seed-0
CIs vs closed-book: LoRA Hebbian +1.8 [−1.8, +5.3], its S=0 +2.3, LoRA-null
S=0 +2.1; vs cf all within [−1.6, +1.0]. Params: gates 1,152, LoRA 3.84 M.

Small banks (banks-v2, 1,024 dev, delta β=α=1 g 0.2, centred): closed-book
33.8, oracle 74.7, counterfactual oracle 21.3 (follow 24.7). Memory 37.8 /
18.4 / 6.7 at N = 1/4/16; shuffled 26.8 / 16.3 / 4.9; counterfactual 29.2 /
15.7 / 4.2 (follow ≤ 0.1). Memory − shuffled: +11.0 [+8.6, +13.5], +2.1
[−0.2, +4.3], +1.9 [+0.1, +3.6]; memory − cf: +8.6 [+6.7, +10.5], +2.6 [+1.1,
+4.3], +2.5 [+1.3, +3.9]. **Pre-registered small-bank criterion met at
N = 16** (both CIs exclude 0), not at N = 4 (shuffled CI includes 0). The
read still cuts EM far below closed-book (6.7 vs 33.8 at N = 16).

V2 pre-registered, d 1024, layer 18 (10.5 M params, 2,199 s): train_seen
40.23, train_heldout 38.48, dev_in 31.45, dev_out 32.81, cf 30.86 (follow 0),
corpus 512/1024/2048 31.84/31.64/30.86; WikiText cloze N 1024 / 4342 / closed
3.91 / 4.10 / 2.15. Probe real/wrong loss: N1 6.879/6.848, N4 5.553/5.631,
N64 2.831/2.810, N1024 3.038/3.042 → no retrieval, as at 1.7B.

V2b final configuration (fewshot 3, standardise, shifted values; 1,000
steps, 1,914 s): probe real/wrong loss and EM: N1 3.055/3.257, 5.86/3.91;
N4 2.860/2.919, 5.47/4.69; N64 2.985/3.024, 4.69/3.52; N1024 3.948/3.668,
2.34/3.12. **Decision rule passes** (gap 0.20 ≥ 0.1 at N = 1; real EM >
wrong EM at N = 64, by 3 of 256 queries). Per the amendment it is evaluated
on TriviaQA (`eval_v2.py`).

## 2026-11-17/18 — 4B V2 widths and expanded keys (`run_4b_sweeps.sh`)

README amendment "4B coverage" written before the runs. Queue log
`runs/delta-rule/queue-4b-sweeps{.part1,.part2,}.out`.

- d 4,096 training ran out of memory twice (whole batch at chunk 512; then
  micro-batch 8 at chunk 512, in the write backward). Peak over 3 steps at
  N 2,048: chunk 256 OOM, 1,024 23.97 GB, 2,048 21.9 GB; 1,024 vs 2,048 give
  equal weights (bias 5.1924e-6 vs 5.1934e-6). Ran with micro-batch 8 +
  chunk 2,048 + expandable segments (README note before results). The failed
  first log is kept at `runs/.../v2/d4096-s0.oom-1117/`.
- My queue watcher used `pgrep -f run_4b_sweeps.sh`, which matched its own
  command line, so it never saw the queue exit (the first OOM sat ~2 h
  unnoticed). Use the PID or a pattern that cannot match the watcher.

V2 (pre-registered recipe, layer 18, seed 0), dev_in / cf (follow) / corpus
512·1024·2048 / train_seen / dev_out; WikiText cloze N1024 / N4342 / closed:
- d 512 (5.25 M): 31.45 / 31.45 (0) / 31.45·31.05·31.64 / 37.89 / 31.64;
  3.91 / 4.88 / 2.15
- d 2,048 (20.98 M): 28.12 / 27.93 (0) / 31.05·29.49·29.30 / 34.77 / 29.30;
  3.91 / 4.10 / 2.15
- d 4,096 (41.95 M): 20.12 / 19.73 (0) / 29.88·27.93·23.24 / 23.05 / 17.77;
  4.49 / 3.52 / 2.15
- CIs vs closed / vs cf: d 512 −3.7 [−7.4, −0.2] / 0.0 [−0.6, +0.6]; d 2,048
  −7.0 [−10.5, −3.7] / +0.2 [−0.8, +1.2]; d 4,096 −15.0 [−19.1, −10.9] /
  +0.4 [−0.4, +1.2].
- Probe real/wrong loss (N 1, 4, 64, 1024): d 512 6.765/6.752, 5.103/5.167,
  2.785/2.785, 3.038/3.040; d 2,048 7.007/6.984, 6.013/6.039, 3.585/3.504,
  3.044/3.035; d 4,096 7.041/7.019, 6.135/6.147, 4.066/4.342, 3.049/3.053.
  Only d 4,096 at N 64 separates (0.28), EM 3.9 vs 3.1 of 256 → not a
  retrieval signal, and it does not show on TriviaQA.

Expanded keys (frozen; gate by train_seen): delta all D pick g 0.1 (grid
edge), dev_in 24.6 / 14.1 / 23.6 / 25.0 at D 512 / 1,024 / 2,048 / 4,096
(D 1,024 dip not explained; not monotone in D); Hebbian all pick g 2 (low
grid edge), dev_in 31.1 / 32.8 / 33.6 / 34.0, cf within 0.6, follow 0.
D 4,096: delta −10.2 [−13.5, −6.8] vs closed, −0.2 vs cf; Hebbian −1.2
[−3.5, +1.0], −0.2 [−1.0, +0.6]. Criterion not met. Same picture as 1.7B.

## 2026-11-18 — Code review, fixes, capacity test (`run_capacity.sh`)

User asked whether memory ≈ swapped memory could be a code bug. Independent
review agent (given code + protocol, not conclusions): no such bug; checked
cf passages differ (512/512, 1024/1024), read fires each decode step under
KV cache, GQA mapping, k⊗v layout, masks, cache keys, V2 labels, scoring.
Found: (1) no BOS in Qwen3 → `skip_first_token` drops the first content
token (verified: `tok("Hello world") == [9707, 1879]`); (2) centred keys but
uncentred queries before DPFP (verified in `delta_qwen.py`); (3) my
"capacity is not the bottleneck" overclaims (184k tokens vs D ≤ 4096).
Also noted: delta β 0.5 global state is recency-dominated; dev_in passages
are spread over corpus positions 24–4335 (verified), so no ordering artefact.

Fixes: `prepend_sink` + `--sink-prefix`; `center_queries` +
`--center-queries`; `--key-expand` in run_banks. Tests 60 pass. README
amendment before running. Queue log `runs/delta-rule/queue-capacity.out`;
CI table in RESULTS. Calibration (N 16, 512 q, memory − shuffled):
1.7B A 0.05 +0.78 / 0.1 −0.59 / 0.2 −0.59; B 0.05 +0.20 / 0.1 +0.98 / 0.2
+0.20; 4B A 0.05 +0.78 / 0.1 +1.95 / 0.2 +2.15; B 0.05 +0.20 / 0.1 +1.17 /
0.2 +0.78 / 0.5 +0.78 (0.5, 1 give ~0 EM elsewhere). Selected 1.7B A 0.05
(edge), B 0.1; 4B A 0.2, B 0.1.

Result: N* = 0 (1.7B A, B), 1 (4B A, B) → more key space does not extend
the signal. 4B B is far less harmful (N 64: 31.4 vs 1.1). 4B A at the old
gate 0.2 with the sink fix: N 1 +10.3 (old +11.0) reproduces; N 16 +0.3
[−1.3, +2.0] vs shuffled (old +1.9 [+0.1, +3.6]) does not → the earlier
"4B keeps signal to N = 16" is not robust; corrected in RESULTS, decisions,
comment draft.

## 2026-11-18 — Does the dropped first token hide an effect?

User asked whether old runs need redoing. 29/512 dev_in passages start with
an answer alias (often the answer recurs later, e.g. "Monotremes …
Monotremata"); WikiText val 15/4,596. `check_first_token.py`: on the other
483 questions memory − cf is within [−1.2, +0.6] for every global run, same
as on all 512 → no rerun needed for this bug. Expanded-key global runs still
carry the uncentred-query mismatch (labelled in RESULTS); the fixed
small-bank capacity test covers the conclusion; a fixed global rerun was
offered to the user as optional.

## 2026-11-18 — Global D 4,096 rerun with both fixes

User asked for the optional rerun. `run_expand_fixed.sh` (sink prefix +
centred queries, frozen, D 4,096; delta β 0.5 gates 0.1/0.3/1/3, Hebbian
β 1/T gates 2/8/32/128), ~1 h, no errors. Selection on train_seen picks the
smallest gate everywhere again (all larger gates give 0 EM). dev_in:
1.7B delta 8.4 (cf 7.2), Hebbian 14.7 (14.8), closed 18.2; 4B delta 29.9
(30.1), Hebbian 31.8 (31.6), closed 35.2. CIs vs closed-book all < 0;
vs cf only 1.7B delta > 0 (+1.2 [+0.4, +2.1]). Criterion not met. Hebbian
got more harmful than the flawed run (query centring raises the read
scale at a fixed gate); 4B delta less harmful. Conclusion unchanged.
`select_v1_expand.py` gained an optional run-root argument.

## 2026-11-18 — 1.7B capacity sensitivity at equal gates

User asked to remove the "1.7B A at grid edge" caveat. Amendment first,
then `run_capacity_sens.sh` (A g0.1, A g0.2, B g0.2; B g0.1 = eval-B),
~45 min, no errors. N*: g0.1 A 1 / B 0; g0.2 A 1 / B 1 → N*(B) ≤ N*(A) at
both gates, same as 4B. A g0.2 N1 +5.4 vs shuffled (cf. old 1.7B +4.5 at
0.2 without the sink fix). Caveat resolved in RESULTS.

## 2026-11-19 — Sanity check: store key→value pairs, read with keys

Maintainer is sceptical of "no gain at all". `sanity_kv.py` tests the store
alone, outside generation. Outputs `runs/delta-rule/sanity/`.

- Synthetic (d 128, random values, top-1 = nearest stored value): iid or
  centred keys 100% up to N 64, delta 97% at N 128 = d, then degrades as
  expected; uncentred correlated keys (mean cos 0.67) fail even at N 16;
  DPFP D 4096 delta 98% at N 2048. Memory code correct.
- Real facts (Qwen3-1.7B, 2048 train questions, value = answer input
  embedding, mean-pooled hidden key). First attempt failed (25% at N 16 even
  with the exact key): the first (sink) token has a massive activation, and
  with mean pooling its 1/length weight made keys encode length (one
  direction 82% of variance, pair cos 0.995). Fixed by excluding token 0 and
  standardising per dimension.
- After the fix, exact key (question written, question read), delta D 8192:
  ~97–100% at every N up to 2048 (chance 0.05%); D 2048 63% at 2048;
  Hebbian collapses with DPFP at large N.
- Passage written, question read (the issue-#6 setting), delta, layer 27:
  N 16 81%, 256 44–48%, 2048 16–21% (chance 0.05%); layer 14 lower.
  Bounded by addressing (question key nearest to its own passage key: 69% /
  72% / 52%).
Reading: the store and frozen Qwen representations do retrieve facts by
key, far above chance even at 2048 facts. The issue-#6 failure must come
from what was used as keys (head-level, per-token) and/or how the read
value is injected into generation; check whether V2's layer-14 inputs had
the same sink-token problem.

## 2026-11-19 — Sanity check 2: does Qwen answer from what the memory returns?

V2 first-token check: the V2 writer skips token 0 (`passage_tokens`,
`skip_first=True`) and writes per token (no pooling), so the pooling/sink
flaw of the first sanity attempt does not apply to V2.

`sanity_inject.py`: value = input embeddings of the answer tokens (8 slots),
key = standardised mean-pooled hidden state (sink excluded), DPFP ν 2
(D 8192), delta. The read is put into Qwen's input as soft tokens after
"Memory:". Qwen3-1.7B, train questions. EM %, layer 27 (layer 14 within ±1,
except passage-key reads, which are lower):

Same 256 questions (seed 0) for both bank sizes, so closed-book and the
ceiling are identical across columns (a first version scored N 2048 on 512
questions, which shifted closed-book to 20.3 — a question-set effect, not the
memory):

| condition | N 256 | N 2048 |
|---|---:|---:|
| closed-book | 21.1 | 21.1 |
| answer typed as text (ceiling) | 64.5 | 64.5 |
| exact stored embeddings (= text, path check) | 64.5 | 64.5 |
| memory, question key, soft | 61.7 | 36.3 |
| memory, question key, snapped to tokens | 62.9 | 54.7 |
| memory, passage key, soft | 25.0 | 18.0 |
| memory, passage key, snapped | 28.1 | 20.7 |
| shuffled memory (question key) | 18.0 | 17.6 |

Outputs `runs/delta-rule/sanity/inject-L{27,14}-same256.json`.

Reading: fill → read by key → give to frozen Qwen works (21 → 62 at 256
facts; shuffled gives no gain, follow ≤ 3.5% because swapped answers do not
fit the question). Qwen is sensitive to blur in soft tokens (snapping helps
at 2048). When the passage is written and the question reads (the issue-#6
design), the gain mostly disappears: the frozen question↔passage key match is
the weak link. Caveat: values here are the answers themselves, easier than
issue #6, where whole passages were stored.

## 2026-11-19 — Is V1 dominated by one key direction?

User asked whether the sink/pooling flaw of the sanity script affects V1. It
does not (V1 writes per-token head keys, no pooling; the separate 2026-11-18
dropped-first-token bug is fixed with `--sink-prefix`). Direct check on V1's
real keys (Qwen3-1.7B, 64 calibration passages with sink prefix, 2,683
tokens, all 28×8 heads; per head: share of variance in the top direction,
effective rank = participation ratio of singular values):

| keys | top-direction share (median / max) | effective rank (median, of 128) | mean abs cos |
|---|---|---:|---:|
| raw | 0.71 / 0.99 | 1.9 | 0.71 |
| centred (as used in all V1 runs) | 0.16 / 0.90 | 17.5 | 0.22 |

Centring removes the dominant direction in most heads but keys still occupy
~17 of 128 dimensions, so the usable room per head is far below 128 (and
some heads stay dominated). Not a bug; a property of Qwen's keys. Cheap
remedy to test: whiten keys and queries per head with calibration
statistics (no training) before writing/reading.

## 2026-11-19 — Upper bound for V1: uncompressed key–value memory, softmax read

README amendment (same date) before eval. `--read knn --knn-top-k 4 --layers
all --sink-prefix`, queue `run_knn_upper.sh`, outputs
`runs/delta-rule/knn-upper/1.7b/`. Development (calibration only):
position-free stored keys swamp attention (garbage output) → right-aligned
RoPE positions (unit test: = passage pasted in context for one passage);
reading only later layers was worse; k = 4 chosen (memory − shuffled at
N 16). Eval, 1,024 questions, EM %, closed-book 17.1, oracle passage 70.9:

| N | memory | shuffled | swapped (follow) | m − shuffled | m − swapped | m − closed | BM25@1 | full ctx |
|---:|---:|---:|---|---|---|---|---:|---:|
| 1 | 67.7 | 16.7 | 13.0 (20.9) | +51.0 [+47.6, +54.4] | +54.7 [+51.6, +57.8] | +50.6 [+47.2, +53.9] | 70.9 | — |
| 4 | 49.7 | 11.4 | 9.0 (9.1) | +38.3 [+35.2, +41.5] | +40.7 [+37.6, +43.8] | +32.6 [+29.3, +36.0] | 67.5 | 68.4 |
| 16 | 32.4 | 9.4 | 5.2 (4.8) | +23.0 [+19.9, +26.2] | +27.2 [+24.3, +30.2] | +15.3 [+11.9, +18.8] | 66.9 | 62.3 |
| 64 | 11.3 | 5.0 | 2.3 (1.8) | +6.3 [+4.2, +8.6] | +9.0 [+7.1, +10.9] | −5.8 [−8.5, −3.0] | 65.1 | 53.8 |
| 256 | 1.8 | 2.2 | 0.8 (0.3) | −0.5 [−1.7, +0.7] | +1.0 [+0.1, +2.0] | −15.3 | 62.4 | 43.1 |
| 1024 | 0.9 | 0.6 | 0.7 (0.0) | +0.3 [−0.5, +1.1] | +0.2 [−0.5, +0.9] | −16.2 | 59.1 | — |

(Pasted answer-swapped passage: follow 32.2.) N* = 64 → pre-registered
reading: Qwen's own query→key matching does find stored content (V1
delta-rule N* was 0–1), so V1's failure is largely compression/blur. But it
helps over closed-book only up to N 16, and at N ≥ 256 Qwen outputs random
entities copied from memory (shuffled N 256: "Mandarina", "Travels"): the
memory content drives the answer, but per-token q·k picks the wrong passage
among many. Lexical retrieval (BM25@1 59.1 at N 1024) and real full context
degrade far less. So frozen matching is a second wall, beyond ~16–64
passages.

## 2026-11-19 — Whitening V1 keys: diagnostic

`--whiten` (per head, `W = (C + ε·mean_eig·I)^(-1/2)` from 256 calibration
passages, applied to centred keys and centred queries). Diagnostic on other
calibration passages, delta rule, D 128; effective rank (median head) and
written / unwritten self-retrieval cosine (value-mean removed):

| N (tokens) | centred | whitened ε 0.01 | ε 0.1 | ε 1 |
|---|---|---|---|---|
| 1 (19) | 7.3; 0.69 / 0.30 | 12.1; 0.89 / 0.35 | 11.8; 0.88 / 0.35 | 10.7; 0.85 / 0.35 |
| 4 (90) | 12.8; 0.56 / 0.29 | 33.3; 0.80 / 0.35 | 31.1; 0.78 / 0.36 | 25.7; 0.72 / 0.35 |
| 16 (512) | 16.6; 0.48 / 0.36 | 73.3; 0.59 / 0.42 | 66.7; 0.58 / 0.41 | 48.5; 0.55 / 0.40 |
| 64 (2,628) | 18.1; 0.42 / 0.38 | 99.1; 0.45 / 0.41 | 88.0; 0.45 / 0.41 | 60.3; 0.44 / 0.40 |

With DPFP ν 16 (D 4,096), gap written − unwritten: centred 0.47 / 0.29,
whitened (ε 0.1) 0.55 / 0.36 at N 4 / 16 (N 64 OOM in this unchunked
diagnostic). Whitening spreads keys (rank ~18 → ~88) and helps while the
memory is not overfull; at N 64 (2.6k tokens ≫ 128) only bigger keys can
help. ε = 0.1 fixed for the QA test (README amendment).

## 2026-11-19 — Whitening V1 keys: QA test (pre-registered)

1.7B, delta β = α = 1, centred + whitened (ε 0.1) keys and queries, sink
fix; W = D 128, WB = DPFP ν 16 (D 4,096). Gate by calibration N 16 memory −
shuffled (`select_capacity_gate.py`):

| gate | 0.05 | 0.1 | 0.2 | 0.5 | 1 |
|---|---|---|---|---|---|
| W mem / shuf | 17.8 / 18.0 | 15.4 / 17.0 | 7.8 / 10.2 | 0.0 / 0.0 | 0.0 / 0.0 |
| WB mem / shuf | 18.8 / 20.1 | 18.0 / 18.8 | 16.2 / 17.4 | 1.6 / 1.0 | 0.0 / 0.0 |

W: memory < shuffled at every non-collapsed gate; the rule picked 0.5
only because 0 − 0 beats the negative differences (degenerate choice; no
gate shows a calibration signal either way). Eval, 1,024 questions, EM %,
closed-book 17.1, paired 95% CIs:

| config | N | memory | shuffled | cf | m − shuffled | m − cf | m − closed |
|---|---:|---:|---:|---:|---|---|---|
| W (g 0.5) | 1 | 6.2 | 6.5 | 5.9 | −0.4 [−2.1, +1.3] | +0.3 [−1.1, +1.7] | −10.9 |
| | 4 | 0.3 | 0.4 | 0.3 | −0.1 | 0.0 | −16.8 |
| | 16, 64 | 0.0 | 0.0 | 0.0 | 0 | 0 | −17.1 |
| WB (g 0.5) | 1 | 17.2 | 10.9 | 12.0 | +6.2 [+4.0, +8.5] | +5.2 [+3.5, +6.9] | +0.1 [−2.1, +2.4] |
| | 4 | 6.9 | 5.4 | 5.4 | +1.6 [−0.1, +3.2] | +1.6 [+0.4, +2.7] | −10.2 |
| | 16 | 1.4 | 1.2 | 0.9 | +0.2 [−0.7, +1.1] | +0.5 [−0.2, +1.2] | −15.7 |
| | 64 | 0.0 | 0.0 | 0.0 | 0 | 0 | −17.1 |

N*(W) = 0, N*(WB) = 1 → criterion (N* > 1) **not met**. Compared with the
capacity test (1.7B A, B: N* = 0), WB at N 1 has a larger content signal
vs controls (+6.2 vs +1.8), but only because the controls hurt more; memory
never beats closed-book. Whitening makes the memory read louder (stronger
gate needed, more damage) without making it more answer-specific. Reading:
spreading the keys fixes the key geometry (rank 18 → 88) but not the
question→passage matching: the question's queries still do not point at the
answer tokens' keys. Consistent with the upper bound (exact uncompressed
keys work to N 16–64): the limit is compression plus frozen matching, and a
linear transform of keys cannot fix the matching. Outputs
`runs/delta-rule/whiten/1.7b/`, queue `run_whiten.sh`.

## 2026-11-19 — Sanity check 3: self-read of the real V1 memory

`sanity_selfread.py`, 1.7B, delta β = α = 1, all 28 layers × 8 KV heads,
sink prefix, centred keys; memory filled as in `run_banks.py` with N eval
passages, read through `_read` (gate 1) with the stored tokens' own keys.
top-1 = right token identified among all stored values (mean over heads);
cos = centred cosine(read, stored value), written / unwritten keys:

| N (tokens) | chance | A (D 128) | B (DPFP 4,096) | W (whitened) | WB (whitened + DPFP) |
|---|---|---|---|---|---|
| 1 (44) | 2.3% | 60%; .63/.31 | 77%; .78/.29 | 88%; .83/.36 | 94%; .87/.31 |
| 4 (175) | 0.6% | 49%; .52/.32 | 72%; .71/.33 | 84%; .71/.38 | 92%; .84/.37 |
| 16 (700) | 0.14% | 36%; .45/.36 | 67%; .63/.40 | 55%; .53/.42 | 92%; .80/.49 |
| 64 (2.8k) | 0.04% | 26%; .43/.40 | 58%; .58/.46 | 33%; .45/.43 | 87%; .72/.53 |
| 256 (11.8k) | — | 18% | 44% | 23% | 68% |
| 1024 (44k) | — | 14% | 35% | 17% | 47% |

Storage with exact keys: the plain V1 memory (A) is lossy even for one
passage (60%), because sequential delta writes with near-parallel keys
overwrite earlier tokens. Whitening + DPFP (WB) stores tokens almost
perfectly up to 64 passages (87%). Yet WB QA had no content signal beyond
N 1 (whitening test above) → for WB, storage is not the wall; the
question → passage matching is.

## 2026-11-19 — Sanity check 4: QA-pair memory (one question → one answer)

`qa_memory`: the memory stores the exact closed-book prompt + gold answer
(README amendment). Gates on calibration (N 1, 512 q): A 0.05/0.1/0.2/0.5/1
→ qa − shuffled +2.3/+3.7/+11.7/+53.5/+6.8 → **0.5**; B → +1.8/+2.5/+7.6/
+5.1/(collapsed) → **0.2**. Eval, 1,024 questions, closed-book 17.1:

| config | N | qa_memory | shuffled_qa | cf_qa (follow) | qa − shuffled | qa − cf | qa − closed |
|---|---:|---:|---:|---|---|---|---|
| A g 0.5 | 1 | **52.4** | 2.9 | 5.2 (10.3) | +49.5 [+46.4, +52.6] | +47.3 [+44.1, +50.5] | **+35.4 [+32.0, +38.7]** |
| | 4–64 | ≤ 0.1 | 0 | 0 | ~0 | ~0 | −17 |
| B g 0.2 | 1 | 20.3 | 13.2 | 14.1 (0.0) | +7.1 [+5.3, +9.0] | +6.2 [+4.5, +8.0] | +3.2 [+1.1, +5.4] |
| | 4 | 13.6 | 12.0 | 11.8 | +1.6 [−0.1, +3.2] | +1.8 [+0.3, +3.3] | −3.5 |
| | 16 | 10.4 | 8.9 | 9.8 | +1.5 [−0.1, +3.0] | +0.6 [−0.7, +1.9] | −6.7 |
| | 64 | 6.8 | 6.4 | 6.1 | +0.4 | +0.8 | −10.3 |

Passage `memory` at N 1 with the same gate: A 8.3 (−8.8 vs closed), B 14.5.
A, N 1, split by closed-book: questions Qwen already knew (175) 81.1% EM;
questions it did not know (849) **46.5%** EM → real new knowledge from
memory. Counterfactual follow is low (10.3%; the swapped answers are
mostly implausible for the question, e.g. "October" for the abacus
question; pasted swapped passage in context: 32.2). At N ≥ 4 with gate 0.5
the output collapses to garbage (".assertIs…"), so the A ladder beyond N 1
says nothing about capacity (the gate was chosen for N 1).

Criterion: works and helps at N 1 for both A and B; N* = 1 for both.
Reading: with the *same* text stored and asked (question tokens = stored
keys), frozen V1 memory gives +35 points over closed-book; with the passage
stored, the same memory and gate give −9. The read and the value path work;
what fails is matching the question to passage-worded keys. Together with
check 3: the next lever is the question→passage key/query matching (trained
retriever / projection), not storage.

## 2026-11-20 — Sanity check 5: read memory with the same passage (cloze)

`sanity_cloze.py` (README amendment 2026-11-20). Memory = banks of N eval
passages written as usual; test prompt = sink + the same passage cut right
before the answer (identical tokens), greedy continuation, hit = starts
with the gold answer. 871 items (prefix ≥ 3 words). No memory 19.3, full
passage in context (copy) 97.9. Gates chosen per N on calibration: A
0.5/0.5/0.05/0.1, B 0.2 all, WB 0.5/0.5/0.2/0.2 (N 1/4/16/64). Hit %, paired
95% CIs:

| config | N | memory | shuffled | cf (follow) | m − shuffled | m − cf | m − none |
|---|---:|---:|---:|---|---|---|---|
| A (D 128) | 1 | 21.9 | 7.9 | 7.8 (0.7) | +14.0 [+11.5, +16.6] | +14.1 [+11.7, +16.5] | +2.6 [−0.1, +5.3] |
| | 4 | 4.0 | 2.4 | 2.1 (0) | +1.6 [+0.3, +3.0] | +2.0 [+0.9, +3.0] | −15.3 |
| | 16 | 18.5 | 17.9 | 18.4 (0) | +0.6 [−0.5, +1.6] | +0.1 | −0.8 |
| | 64 | 13.1 | 12.2 | 12.3 (0) | +0.9 [−0.5, +2.3] | +0.8 | −6.2 |
| B (DPFP 4,096) | 1 | 20.7 | 15.7 | 16.9 (0) | +4.9 [+3.2, +6.8] | +3.8 [+2.5, +5.3] | +1.4 [−0.3, +3.2] |
| | 4 | 17.3 | 15.0 | 15.7 (0) | +2.3 [+0.7, +3.9] | +1.6 [+0.3, +2.9] | −2.0 |
| | 16 | 15.6 | 14.6 | 14.2 (0) | +1.0 [−0.5, +2.5] | +1.4 [+0.3, +2.5] | −3.7 |
| | 64 | 13.1 | 14.0 | 12.9 (0) | −0.9 | +0.2 | −6.2 |
| WB (whitened + DPFP) | 1 | 20.9 | 10.8 | 14.4 (0) | +10.1 [+7.9, +12.4] | +6.5 [+4.8, +8.4] | +1.6 [−0.5, +3.7] |
| | 4 | 11.4 | 7.7 | 9.0 (0) | +3.7 [+2.1, +5.4] | +2.4 [+1.1, +3.7] | −7.9 |
| | 16 | 17.8 | 17.3 | 16.5 (0) | +0.5 [−0.9, +1.8] | +1.3 [+0.2, +2.4] | −1.5 |
| | 64 | 14.9 | 15.3 | 15.8 (0) | −0.3 | −0.9 | −4.4 |
| K (softmax, uncompressed) | 1 | **98.6** | 19.7 | 5.2 (89.0) | +78.9 [+76.1, +81.6] | +93.5 | +79.3 [+76.6, +82.1] |
| | 4 | **92.0** | 16.9 | 5.4 (75.4) | +75.1 | +86.6 | +72.7 |
| | 16 | **85.4** | 15.4 | 4.9 (65.4) | +70.0 | +80.5 | +66.1 |
| | 64 | **64.6** | 13.0 | 3.6 (45.5) | +51.7 | +61.1 | +45.4 [+41.7, +49.0] |

N*: A 4, B 4, WB 4, K 64. "Helps" (m − none CI > 0): delta-rule configs at
no N; K at every N. Examples (passage "…producer ___ themselves call…"):
none ", the late John L. Sullivan", A ", are the main characters", K
"Richard D'Oyly Carte themselves call", K with swapped memory "Third
Crusade themselves call".

**Reading — corrects the check-4 note above.** With perfect query = key
match (same passage, same tokens), the softmax memory reproduces the
passage (98.6% at N 1, 64.6% at N 64, and follows swapped content), but
the delta-rule memory does not help in any config: it beats the controls
mainly because the controls hurt, never no-memory, and never copies a
swapped answer (follow 0). So for passages the delta-rule read itself
(compressed state + linear read) is a wall *independent of matching*;
"matching is what fails" (check 4) was too strong. Check 4's QA-pair
success (+35 at N 1) is then the special case, not the rule; why it works
there is open (hypothesis, untested: the chat-template tokens before the
answer are highly distinctive keys, so the read at that position returns
the answer tokens almost alone).

## 2026-11-20 — Variant 1T (trained feature maps, delta rule β 1): result

Training (`runs/delta-rule/featmap-train.out`, 3,000 steps, ~30 min):
validation relative error (1 = reading nothing) fell from 1.17 / 1.15 /
1.23 / 1.52 (N 1 / 4 / 16 / 64, identity maps) to **0.89 / 0.80 / 0.74 /
0.70** and had plateaued from step ~2,000. By layer at N 1: 0.77–0.99
(late layers ≈ 0.98); at N 64: 0.59–0.87.

Eval, gate 1 (pre-registered), closed-book 17.1:

| N | QA memory | shuffled | cf | cloze memory | shuffled | cf (follow) |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.1 | 1.9 | 3.0 | 9.0 | 8.3 | 8.0 (0) |
| 4 | 0.0 | 0.0 | 0.0 | 3.9 | 3.8 | 3.9 (0) |
| 16 | 0.0 | 0.0 | 0.0 | 1.0 | 0.9 | 1.0 (0) |
| 64 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 (0) |

(cloze: no memory 19.3.) Criterion: N\*(T) = 0, does not help → **fails**.
At N ≥ 4 QA answers are empty strings: the read, added at full strength in
all 28 layers, still carries 70–90% error and breaks the forward pass.

Why the distillation plateaus (diagnosis, not tested yet): the target
`o_joint − o_plain` equals `λ (o_mem − o_plain)`, with λ = the share of
softmax mass the memory takes from the context. It depends on the
context (o_plain, the context normaliser), which an additive read
`phi_q(q)ᵀ S` never sees, so most of the target is unlearnable in this
form. Hedgehog avoids this by normalising memory and context jointly. Fix
to test next: output = (1 − λ̂) o_plain + λ̂ ô_mem, ô_mem = delta-rule read
(normalised), λ̂ = m / (m + Z_ctx) with m = phi_q(q)·z (z = Σ phi_k(k)) and
Z_ctx the context softmax normaliser; plus learned per-token β.

## 2026-11-23 — Sanity check 6: usual question, memory read with the passage's mean query

README amendment of 2026-11-23. `run_passage_query.sh`, Qwen3-1.7B, banks-v2
eval (1,024 questions; closed-book 17.1). Every layer reads the memory with the
mean pre-RoPE query of the question's own passage (counterfactual: of the
stored swapped passage) at every token, not the token's own query. Paired 95%
bootstrap CIs (10,000 resamples, seed 20261111); "follow" = answers the
swapped answer. Script: `ci/pq_ci.py`.

| Config | N | gate | memory | shuffled | counterf. (follow) | mem − shuf | mem − cf | mem − closed |
|---|---|---|---|---|---|---|---|---|
| K (softmax, top-k 4) | 1 | – | 0.0 | 0.0 | 0.0 (0.0) | +0.0 | +0.0 | −17.1 [−19.4, −14.7] |
| K | 4 | – | 5.9 | 6.2 | 4.6 (0.0) | −0.3 [−1.4, +0.8] | +1.3 [+0.3, +2.2] | −11.2 |
| K | 16 | – | 5.7 | 5.6 | 4.7 (0.0) | +0.1 | +1.0 | −11.4 |
| K | 64 | – | 1.6 | 1.9 | 1.3 (0.0) | −0.3 | +0.3 | −15.5 |
| A | 1 | 0.2 | **19.9** | 16.2 | 16.9 (0.0) | +3.7 [+1.8, +5.7] | +3.0 [+1.6, +4.6] | **+2.8 [+0.9, +4.8]** |
| A | 4 | 0.2 | 9.4 | 10.3 | 9.5 (0.0) | −0.9 | −0.1 | −7.7 |
| A | 16 | 0.05 | 17.1 | 16.7 | 17.1 (0.0) | +0.4 | +0.0 | +0.0 |
| A | 64 | 0.05 | 4.9 | 4.9 | 4.9 (0.0) | +0.0 | +0.0 | −12.2 |
| B | 1 | 0.2 | 14.5 | 12.7 | 12.6 (0.0) | +1.8 [+0.1, +3.5] | +1.9 [+0.5, +3.2] | −2.6 [−4.7, −0.6] |
| B | 4 | 0.2 | 13.0 | 12.7 | 13.1 (0.0) | +0.3 | −0.1 | −4.1 |
| B | 16 | 0.2 | 10.6 | 10.3 | 11.8 (0.0) | +0.4 | −1.2 | −6.4 |
| B | 64 | 0.05 | 6.4 | 6.7 | 6.5 (0.0) | −0.3 | −0.1 | −10.6 |
| WB | 1 | 0.5 | 17.2 | 13.6 | 13.4 (0.0) | +3.6 [+1.8, +5.6] | +3.8 [+2.3, +5.3] | +0.1 [−2.1, +2.3] |
| WB | 4 | 0.5 | 10.4 | 9.1 | 8.6 (0.0) | +1.4 [−0.5, +3.1] | +1.9 [+0.8, +3.0] | −6.6 |
| WB | 16 | 0.1 | 17.5 | 18.2 | 17.7 (0.0) | −0.7 | −0.2 | +0.4 |
| WB | 64 | 0.05 | 6.3 | 6.2 | 6.1 (0.0) | +0.2 | +0.3 | −10.7 |

N\*: K 0, A 1, B 1, WB 1 (the same as with the question's own query). By the
pre-registered criterion, reading with the passage gives a content signal only
at N = 1. It helps (memory − closed-book CI > 0) only for A at N = 1: +2.8
points, versus +0.9 with the question's own query (capacity test, A N = 1:
18.0). Nobody ever answers the swapped answer (follow 0.0 in every cell). So
the small N = 1 gain is not Qwen reading the stored answer: the memory
nudges Qwen towards the right topic, and the swapped memory does not push it
to the swapped answer. From N = 4 on, every delta-rule config is below
closed-book (the same mean read added at every token degrades the output),
except where the calibration picked the smallest gate (≈ no-op).

K (uncompressed softmax) breaks completely: at N = 1 every answer is empty.
The mean query matches the same few stored tokens strongly at every position,
so they take a large share of every token's attention away from the
question. The softmax memory works only with the token's own query (QA N = 1
upper bound, cloze 98.6%); one averaged query for all tokens is the wrong
query for it.

Reading: knowing exactly which passage to read does not rescue the
delta-rule memory. Retrieval by the passage removes the question-vs-passage
mismatch, and still at most ~3 points come back at N = 1, with no sign of
the stored answer itself being read.

Why A scores 9.4 / 17.1 / 4.9 at N = 4 / 16 / 64 (not "16 better than 4"):
the cells use different gates (0.2 / 0.05 / 0.05). The rule (max memory −
shuffled on calibration) picked 0.2 at N = 4 for a noise-sized lead (+1.4 on
512 questions) although it costs ~10 points; at N = 16, gates ≥ 0.2 already
broke Qwen, so only 0.05 (memory almost off → closed-book level) was left.
Calibration EM of A at the same gate 0.05: N 4 / 16 / 64 = 19.9 / 19.1 / 5.1,
i.e. falling with N (the read-out grows with the bank, so a fixed gate
pushes harder). The rule rewards noise-sized leads at harmful gates; it
does not change N\*.

Fixed-gate row (README amendment "fixed-gate row"; gate 0.05 for every N;
`run_passage_query_fixed.sh`, outputs `runs/delta-rule/passage-query/1.7b/fixed-g0.05/`,
`ci/pq_fixed_ci.py`). EM %, closed-book 17.1, follow 0.0 everywhere:

| Config | N 1 mem / shuf / cf | N 4 | N 16 | N 64 |
|---|---|---|---|---|
| A | 18.2 / 17.7 / 17.2 | 17.9 / 17.3 / 17.5 | 17.1 / 16.7 / 17.1 | 4.9 / 4.9 / 4.9 |
| B | 18.1 / 17.0 / 17.5 | 17.5 / 16.5 / 17.7 | 16.4 / 17.1 / 16.3 | 6.4 / 6.7 / 6.5 |
| WB | 17.9 / 17.0 / 17.8 | 17.3 / 17.6 / 17.5 | 17.1 / 17.8 / 17.6 | 6.3 / 6.2 / 6.1 |

At one gate the scores fall monotonically with N for every config (A 18.2 →
17.9 → 17.1 → 4.9). The memory's lead over both controls has a CI above 0
nowhere (largest: A N 1 vs cf +1.0 [+0.2, +1.8] but vs shuffled +0.5
[−0.4, +1.4]); A at N 1 is +1.1 [+0.1, +2.1] over closed-book. At N 64 the
gate-0.05 read already costs ~11 points, independent of what is stored.

### 2026-11-24 — Check 6: code verification and why it fails

Wiring (`ci/pq_wiring.py`, real Qwen3-1.7B, `run_banks.run_bank`, 4
questions): all 112 memory reads (28 layers × prompt pass + 3 decode steps)
used each row's own passage mean query (max difference 0.06 = bf16
rounding; different passages' queries differ by up to 1.5); `read_queries`
is reset afterwards. Unit tests cover the masked mean and the knn rotation.

What the read returns (`ci/pq_what.py`, config A, gate 1, first 64
eval passages, mean over all layers/heads). answer_rank = percentile of the
answer tokens' values among the own passage's values by cosine to the
read-out (0.5 = chance); soft = the same query in a softmax over the stored
(pre-RoPE) keys; answer tokens are 7.4% (N 1) / 6.6% (N 4) of stored tokens.

| N | query | answer_rank | own passage top-1 (chance) | soft: answer share | soft: own-passage share |
|---|---|---|---|---|---|
| 1 | passage mean | 0.52 | 1.00 (1.00) | 0.070 | 1.00 |
| 1 | question last token | 0.54 | 1.00 | 0.075 | 1.00 |
| 1 | token before the answer | 0.58 | 1.00 | 0.085 | 1.00 |
| 4 | passage mean | 0.50 | 0.40 (0.25) | 0.019 | 0.28 |
| 4 | question last token | 0.51 | 0.28 | 0.020 | 0.26 |
| 4 | token before the answer | 0.55 | 0.35 | 0.027 | 0.30 |

Reading: the passage mean query finds the right passage somewhat better at
N 4 (40% vs 28% with the question's query, chance 25%), but no query brings
back the answer: the read-out is a blend of the whole passage (answer rank ≈
chance), and the answer is ~7% of what is stored. At N 1 "which passage" is
trivially solved for any query (only one is stored), yet the memory still did
not help in check 2, so the question-vs-passage direction mismatch is not
the main failure; picking the answer *within* the passage is. A mean query
means "everything in this passage", so even a perfect read would return the
passage's average content. In Qwen this selection is done by softmax over
the context with each token's own query (cloze K 98.6%), which a single
averaged linear read cannot do.

## 2026-11-24 — V1-LoRA on small banks: result

README amendment of 2026-11-24, `run_lora_banks.sh`, Qwen3-1.7B, config A
(delta β 1, centred keys, D 128), writing by frozen Qwen; LoRA r8 q/o + gates
(1.84 M), 2,000 steps, final checkpoint. Validation loss (N 1 / 4 / 16):
LoRA 1.61 / 3.31 / 3.65 → 0.89 / 1.74 / 2.35 (best ~step 750–1250, slight
overfit after); control 2.53 / 3.46 / 4.07 → 1.64 / 1.98 / 2.13. Mean gate:
LoRA 0.1 → 0.037; control 0.1 → −0.0005 (it learns to switch memory off).
Paired CIs as before (`ci/lora_ci.py`). Frozen closed-book 17.1.

| LoRA | N | memory | shuffled | counterf. (follow) | mem − shuf | mem − cf | mem − closed (same LoRA) | mem − control LoRA mem |
|---|---|---|---|---|---|---|---|---|
| memory-trained | closed-book | 20.4 | | | | | | |
| | 1 | **46.1** | 17.6 | 15.0 (0.5) | +28.5 [+25.6, +31.5] | +31.1 [+28.2, +34.0] | +25.7 [+22.8, +28.6] | +27.0 [+23.8, +30.1] |
| | 4 | **28.7** | 18.7 | 18.4 (0.0) | +10.1 [+7.7, +12.4] | +10.4 [+8.3, +12.4] | +8.3 [+6.1, +10.6] | +9.9 [+7.2, +12.5] |
| | 16 | 21.0 | 18.4 | 18.9 (0.0) | +2.6 [+0.6, +4.7] | +2.1 [+0.6, +3.5] | +0.6 [−1.5, +2.7] | +2.3 [0.0, +4.8] |
| | 64 | 17.7 | 17.7 | 16.1 (0.0) | +0.0 | +1.6 [+0.2, +2.9] | −2.7 [−4.9, −0.6] | −1.2 |
| control (unrelated memory) | closed-book | 19.5 | | | | | | |
| | 1 / 4 / 16 / 64 | 19.1 / 18.8 / 18.7 / 18.8 | 19.7 / 18.6 / 18.3 / 18.7 | 20.3 / 19.4 / 18.8 / 19.1 (0) | all CIs include 0 | | | |

N\*: memory-trained LoRA 16, control 0. Pre-registered criterion, literally:
N\* = 16 > 1 holds; at N\* = 16 memory − control-LoRA memory is +2.3 with
CI lower bound exactly 0.0 (not > 0), and memory − closed-book is not > 0.
So at N\* = 16 "works" misses by the CI edge and "helps" fails. At N = 4 all
three conditions hold with large margins (+10.1, +10.4 vs controls; +9.9 vs
the control LoRA; +8.3 vs its own closed-book), and at N = 1 the gain is
+25.7 points over its own closed-book.

Not fine-tuning: the control LoRA (same data, same steps, unrelated
memory) gains only +2.4 closed-book and its memory does nothing; it drives
its gates to 0. Not prior knowledge only: on questions frozen Qwen gets
wrong (849), memory 38.9% vs shuffled 10.0% at N 1 (20.4 vs 9.9 at N 4); on
known ones 81.1 vs 54.3. Leakage: eval questions and answers share nothing
with the training pool (0 identical questions, 0 shared aliases); 10 eval
passages also occur in the pool (other questions); without them N 1 46.2 /
17.8 / 15.2, N 4 28.9 / 18.7 / 18.4. Examples (N 1, frozen wrong → memory
right): Burslem club "Port Vale" (alone: Wolverhampton), Fayed brothers
"Harrods" (alone: B&Q), mauveine "quinine", Thomas Hardy "architect".

Counterfactual: follow ≈ 0 (0.5% at N 1), but accuracy drops below
closed-book (15.0 vs 20.4): the swapped memory removes the right answer
without making Qwen say the swapped one. Caveat: the swapped-in answers come
from random donor questions and are usually the wrong type (e.g.
"liechtenstein" for a national park, "works of stephen king" for a
profession), so a trained reader can reject them; follow is a weak test
here.

Reading: application-side LoRA trained on small banks makes the frozen-head
delta-rule memory readable, strongly at N 1–4, fading by 16, gone at 64
(capacity of a 128 × 128 state per head with β 1). This is the first
Variant 1 result with a content signal beyond N = 1. The earlier V1-LoRA
failed because it was trained on the ill-conditioned global state.

## 2026-11-24 — V1-LoRA + learned β (option 1, per layer × KV head): result

`run_lora_beta.sh`; outputs `runs/delta-rule/lora-banks/1.7b/{lora-beta-*.pt,eval-beta-*}`;
`ci/lora_beta_ci.py`. Learned β (mean over KV heads) stays ~0.85–0.92 in
layers 0–18 and drops to 0.53–0.73 in layers 19–26 (min 0.08); control's β
mean 0.87 and its gates go to 0.0001 (memory switched off).

| β-LoRA | N | memory | shuffled | counterf. (follow) | mem − shuf | mem − cf | mem − closed (17.5) | mem − control LoRA | mem − fixed-β LoRA |
|---|---|---|---|---|---|---|---|---|---|
| | 1 | **51.4** | 14.8 | 12.7 (1.1) | +36.5 [+33.4, +39.6] | +38.7 [+35.7, +41.7] | +33.9 [+30.8, +37.0] | +32.6 [+29.4, +35.8] | +5.3 [+3.0, +7.5] |
| | 4 | **34.3** | 15.3 | 16.0 (0.1) | +18.9 [+16.3, +21.6] | +18.3 [+15.8, +20.8] | +16.8 [+14.2, +19.4] | +15.2 [+12.4, +18.1] | +5.6 [+3.2, +7.8] |
| | 16 | **22.7** | 17.5 | 17.5 (0.0) | +5.2 [+3.1, +7.2] | +5.2 [+3.5, +6.8] | +5.2 [+2.8, +7.5] | +3.9 [+1.5, +6.3] | +1.7 [−0.1, +3.4] |
| | 64 | 15.9 | 15.3 | 15.1 (0.0) | +0.6 | +0.8 | −1.6 | −1.1 | −1.8 [−3.5, −0.1] |
| control | 1 / 4 / 16 / 64 | 18.8 / 19.0 / 18.8 / 17.0 | 19.4 / 19.0 / 19.1 / 17.8 | 19.9 / 18.8 / 19.0 / 17.6 | all ≤ 0 | | closed 19.0 | | |

Pre-registered criterion ("learned β helps" = at N 16 or 64, memory beats
shuffled and cf AND beats the fixed-β LoRA with CI > 0): **not met** (N 16:
+1.7 [−0.1, +3.4]; N 64: −1.8). Learned β does improve N 1 and N 4 by
+5.3 / +5.6 points (CIs > 0), and with it the previous amendment's criteria
now hold at N\* = 16: memory beats the control LoRA (+3.9 [+1.5, +6.3]) and
its own closed-book (+5.2 [+2.8, +7.5]), i.e. "LoRA makes the memory work
and helps" up to 16 passages. Per the user's plan, option 2 (per-token β)
is next.

## 2026-11-24 — V1-LoRA + per-token β (option 2): result

`run_lora_tbeta.sh`; outputs `runs/delta-rule/lora-banks/1.7b/{lora-tbeta-*.pt,eval-tbeta-*}`;
`ci/lora_tbeta_ci.py`. Closed-book with this LoRA, no memory: 20.2.

| N | memory | shuffled | counterf. (follow) | mem − shuf | mem − cf | mem − closed | mem − control LoRA | mem − fixed-β LoRA | mem − option 1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **54.1** | 18.4 | 15.9 (3.0) | +35.7 [+32.7, +38.9] | +38.2 [+35.3, +41.2] | +33.9 [+30.9, +37.0] | +33.0 [+29.8, +36.3] | +8.0 [+5.2, +10.8] | +2.7 [−0.1, +5.5] |
| 4 | **41.4** | 16.9 | 15.8 (0.8) | +24.5 [+21.7, +27.5] | +25.6 [+22.9, +28.3] | +21.2 [+18.3, +24.1] | +21.0 [+18.1, +24.0] | +12.7 [+9.8, +15.6] | +7.1 [+4.4, +10.0] |
| 16 | **26.5** | 17.1 | 17.0 (0.1) | +9.4 [+7.0, +11.8] | +9.5 [+7.2, +11.8] | +6.2 [+3.8, +8.8] | +5.7 [+3.0, +8.3] | +5.5 [+2.8, +8.1] | +3.8 [+1.4, +6.3] |
| 64 | 15.9 | 14.8 | 13.3 (0.0) | +1.1 [−0.9, +3.0] | +2.6 [+1.0, +4.4] | −4.3 [−6.5, −2.1] | −3.9 | −1.8 [−4.0, +0.5] | +0.0 |

Control (unrelated memory, per-token β): memory 21.1 / 20.4 / 20.8 / 19.8,
every CI vs its shuffled / cf / closed-book (20.9) includes 0; its gates go
to −0.002 (memory off).

Pre-registered criterion **met at N = 16**: memory beats shuffled and cf
(+9.4, +9.5) and the fixed-β LoRA (+5.5 [+2.8, +8.1]). N\* = 16; memory
helps vs its own closed-book up to N 16 (+6.2). N 64: still no signal.

What β learned (200 eval passages, 9,206 tokens): layers 0–12 write
almost everything (mean β 0.9–0.95), later layers much less (0.13–0.4 in
layers 15–24). Within heads β varies by token (median std over tokens 0.14,
max 0.43). The most selective head (layer 24, KV head 0) on the Hawksmoor
passage: "Hawksmoor" 0.95–0.99, "novel" 1.00, "Peter Ackroyd" 0.93–1.00,
"5" 0.97, but "is" 0.02, "by" 0.00, "the" 0.09, "English" 0.06, "writer"
0.04: it writes names and content words and skips function words.
Averaged over all heads, answer tokens are not singled out (0.62 vs 0.60).

## 2026-11-24 — per-token β + bigger memory (config B, DPFP ν 16): result

Pre-registered in the README amendment "per-token β + bigger memory". Same
recipe as option 2, memory keys 128 → 4,096 wide (centred keys/queries +
DPFP ν 16). Qwen3-1.7B, 1,024 eval questions, one seed; frozen closed-book
17.1; this LoRA with memory off 19.8 (control LoRA 19.6). Paired bootstrap
CIs (10,000, seed 20261111). Script: `ci/lora_dtbeta_ci.py`.

| N | memory | shuffled | cf (follow) | m − shuf | m − cf | m − own closed-book | m − control-LoRA mem | m − option 2 (D 128) |
|---|---|---|---|---|---|---|---|---|
| 1 | 53.2 | 14.2 | 13.9 (4.1) | +39.1 [+35.8, +42.4] | +39.4 | +33.4 | +33.8 | −0.9 [−3.7, +2.0] |
| 4 | 41.2 | 15.2 | 15.6 (0.8) | +26.0 | +25.6 | +21.4 | +22.0 | −0.2 [−3.0, +2.5] |
| 16 | 27.0 | 15.1 | 15.8 (0.1) | +11.8 | +11.1 | +7.1 [+4.7, +9.7] | +8.4 | +0.5 [−2.1, +3.1] |
| 64 | 18.9 | 15.7 | 15.5 (0.0) | +3.2 [+1.1, +5.5] | +3.4 [+1.7, +5.2] | −0.9 [−3.1, +1.4] | −0.2 [−2.6, +2.2] | +3.0 [+0.9, +5.2] |

- **Pre-registered criterion: met** (N = 64: memory > shuffled and > cf, and
  > option 2, all CIs > 0). N\* = 64 (option 2: 16).
- **But at 64 the memory does not help answer:** 18.9 is no better than the
  same LoRA without memory (19.8) or the control LoRA with memory (19.1).
  Any 64-passage memory costs about 4 points (shuffled 15.7 vs 19.8); the
  right passages win back about 3 of them. The bigger memory is less noisy
  than D 128 at 64 (option 2 memory 15.9), not yet useful.
- N ≤ 16: no difference from option 2 (CIs span 0). The bigger memory's
  gain is only at 64.
- Control LoRA: gates → 0.0001, no memory effect at any N (as before).
- Training never saw N = 64 banks (sizes 1, 4, 16 as in option 2).

## 2026-11-24 — sanity check: passages in the context instead of memory

README amendment "passages in the context instead of memory". Same 1,024
eval questions and banks-v2 banks; memory off. Frozen numbers from the
original stage (`runs/delta-rule/eval-baselines`); the frozen oracle rerun
reproduces them exactly (1,024 / 1,024 identical predictions). LoRA = the
option 2 (per-token β) LoRA, memory off. Script: `ci/context_ci.py`.

| N | context, frozen | context, LoRA | memory: fixed-β LoRA | option 2 | option 2 + bigger memory | option 2 memory − LoRA context |
|---|---|---|---|---|---|---|
| 1 (own passage) | 70.9 | 67.3 | 46.1 | 54.1 | 53.2 | −13.2 [−16.7, −9.7] |
| 4 | 68.4 | 67.9 | 28.7 | 41.4 | 41.2 | −26.5 [−30.0, −22.9] |
| 16 | 62.3 | 67.9 | 21.0 | 26.5 | 27.0 | −41.4 [−44.9, −37.8] |
| 64 | 53.8 | 64.7 | 17.7 | 15.9 | 18.9 | −48.8 [−52.2, −45.4] |

Closed-book: frozen 17.1, LoRA 20.2.

- Passages in context beat memory at every N, and the gap grows with N.
  Context loses only 6 points from 1 to 64 passages (LoRA: 67.3 → 64.7);
  the best memory loses 35 (54.1 → 18.9).
- At N = 1 the best memory reaches about 80% of the context score (54.1 /
  67.3); at N = 16 about 40%; at N = 64 nothing above closed-book.
- The LoRA (trained only on memory reads) also reads long contexts better:
  LoRA − frozen +5.6 [+2.8, +8.3] at 16, +10.9 [+8.0, +13.9] at 64, F1 61.1 →
  72.7 at 64; not a format effect (same answer length, different and correct
  answers). At N = 1 it is slightly worse (−3.6 [−6.3, −0.8]).
- Answer-swapped passage in context: frozen follows it 32.2% (EM on the true
  answer 7.4); LoRA follows 21.2% (EM 16.2). Memory follow rate at N = 1:
  3.0%. Qwen trusts a passage it can see far more than one in memory.

## 2026-11-25 — Variant 2 on small banks (step A): result

README amendment "Variant 2 on small banks (step A)". `eval_v2_banks.py`,
banks-v2 eval, 1,024 questions; paired bootstrap CIs (10,000, seed
20261111; `ci/v2banks_ci.py`). Closed-book here 16.5 (plain) /
21.1 (+ LoRA) instead of 17.1 / 20.2 because this script keeps
`eval_v2.py`'s TF32 setting; every comparison below is within one run.

| run | closed-book | memory N = 1 / 4 / 16 / 64 | shuffled | counterfactual | N\* (retrieves) |
|---|---|---|---|---|---|
| V2b final (d 1024) | 16.5 | 16.1 / 15.7 / 13.9 / 13.2 | 15.5 / 14.2 / 12.2 / 13.8 | 14.9 / 14.6 / 14.0 / 12.3 | 0 |
| V2b + V1 LoRA | 21.1 | 17.5 / 17.0 / 16.4 / 16.3 | 17.8 / 17.3 / 16.0 / 15.7 | 17.0 / 15.6 / 16.2 / 15.4 | 0 |
| V2 d 512 | 16.5 | 19.6 / 19.8 / 18.8 / 17.2 | 19.5 / 19.5 / 19.0 / 17.5 | 19.1 / 19.7 / 19.4 / 17.0 | 0 |
| V2 d 1024 | 16.5 | 18.7 / 20.3 / 19.9 / 18.8 | 18.4 / 20.1 / 19.9 / 18.8 | 18.6 / 20.2 / 19.8 / 18.7 | 0 |
| V2 d 2048 | 16.5 | 17.3 / 19.7 / 20.4 / 18.1 | 17.4 / 20.0 / 20.3 / 18.6 | 17.9 / 19.6 / 20.3 / 18.4 | 0 |
| V2 d 4096 | 16.5 | 17.4 / 19.2 / 21.3 / 18.8 | 17.3 / 19.2 / 20.7 / 19.3 | 17.6 / 18.9 / 21.3 / 18.8 | 0 |
| V2 d 512–4096 + V1 LoRA | 21.1 | 16.4–21.6 | same as memory ± 0.7 | same ± 0.9 | 0 |

- **No Variant 2 interface retrieves on small banks**, with or without the
  LoRA: memory never beats both shuffled and counterfactual (largest
  single difference +1.7; never both CIs > 0 at one N). Follow rate 0
  everywhere.
- The pre-registered V2 (d 512–4096) gives +2 to +5 over closed-book with
  *any* memory (right, unrelated or answer-swapped alike): the
  content-independent prior already seen in its positive control, not
  retrieval. V2b (the version with the prior removed) is at or below
  closed-book.
- The V1 LoRA adds nothing to V2: with it every V2 memory is at or below
  the LoRA's own closed-book (21.1), and harms at N ≥ 16.

## 2026-11-25 — Variant 2 step B (writer + reader from Qwen's heads, WikiText-trained): result

README amendment "Variant 2, step B". `train_lora_banks.py --task cloze
--writer-rank 8 --learn-beta token` (WikiText fill-in-the-blank only; no
TriviaQA in training), then `run_banks.py` config A on banks-v2 eval,
1,024 questions; paired bootstrap CIs (10,000, seed 20261111;
`ci/v2q_ci.py`). Plain frozen Qwen closed-book on these questions: 17.1.

| run | closed-book (same model) | memory N = 1 / 4 / 16 / 64 | shuffled | counterfactual (follow) | N\* |
|---|---|---|---|---|---|
| B main (control none) | 7.8 | 48.5 / 25.0 / 10.5 / 4.0 | 2.7 / 0.9 / 1.3 / 2.6 | 1.7 / 0.4 / 1.8 / 2.0 (27.6 / 9.4 / 0.8 / 0.0) | 64 |
| B control model (trained with unrelated memory) | 22.7 | 23.4 / 22.7 / 22.1 / 21.9 | 22.3 / 21.9 / 21.7 / 21.6 | 23.3 / 22.8 / 22.1 / 22.0 | 0 |
| B main + V1 LoRA stacked | 10.3 | 47.5 / 31.2 / 15.7 / 7.6 | 7.4 / 4.4 / 5.4 / 4.5 | 5.4 / 3.7 / 5.9 / 4.7 (14.6 / 4.5 / 0.5 / 0.0) | 64 |

Pre-registered criteria (main model):
- **Retrieves** at every N (memory − shuffled and memory − cf CIs > 0;
  at N = 64 only just: +1.4 [+0.1, +2.6], +2.1 [+1.0, +3.2]).
- **Works**: memory − control-model memory +25.1 [+21.4, +28.9] at N = 1;
  +2.3 [−1.1, +5.8] at N = 4 (n.s.); negative at 16 / 64.
- **Helps** (memory − own closed-book CI > 0) at N = 1 / 4 / 16
  (+40.7, +17.2, +2.7 [+0.5, +5.0]); not at 64 (−3.8).
- Not pre-registered, reported for honesty: the trained reader damages
  closed-book recall (7.8 vs 17.1 plain Qwen; answers become one-word,
  cloze-style). Against plain Qwen's closed-book, memory is +31.4 / +7.9
  at N = 1 / 4 and −6.5 / −13.1 at 16 / 64. Stacking the V1 LoRA:
  +30.4 / +14.1 / −1.4 (n.s.) / −9.5.
- WikiText val loss (main): right memory 0.40 / 0.63 / 2.21 vs wrong
  5.85 / 4.87 / 4.37 at N = 1 / 4 / 16 sentences. The control model
  learned to ignore memory (gate mean 0.014; right = wrong).
- First Variant 2 that retrieves on TriviaQA. Compared with Variant 1's
  best (per-token β + DPFP, TriviaQA-trained: 53.2 / 41.2 / 27.0 / 18.9)
  it is close at N = 1 and falls off faster with N.
