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
