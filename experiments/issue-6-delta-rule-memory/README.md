# Issue #6: delta-rule memory on TriviaQA

GitHub issue: https://github.com/Necr0x0Der/memWMLM/issues/6

Branch: `delta-rule`. Status: **protocol draft, local only** (not yet posted to
the issue). This file is the local experiment record required by
`for_agents/README.md` until it is imported into the issue.

This protocol was written before inspecting any memory-condition results.

## Claim (one at a time)

**C1 (primary).** A write-once associative matrix memory `S`, built by the delta
rule from a bank of `N` TriviaQA evidence passages and *without* those passages in
the application context, lets frozen Qwen3-1.7B answer the bank's questions with
exact match materially above closed-book and `null_memory`, approaching
BM25-RAG / full-context on the same bank; and the gain is causal (it collapses
under shuffled memory and follows counterfactual answer swaps).

C1 is tested separately for **Variant 1** (existing Qwen heads) and **Variant 2**
(trained writer/reader interface). A positive result for one variant does not
transfer to the other.

## Design decisions (and why)

### D1 — Fact unit: short evidence passages, not raw documents and not QA pairs

TriviaQA RC (Wikipedia domain) provides evidence documents per question. A fact
unit is a short passage (6–64 words: one sentence, or the previous plus the
answer sentence so that leading pronouns resolve) taken from the question's
Wikipedia evidence, whose final sentence contains an answer alias. Among
candidates, the one with most content-word overlap with the question wins (ties:
shorter, then earlier). Rule `max_question_overlap_answer_sentence_or_window_v2`
in `prepare_banks.py`; the matched answer span is stored for the counterfactual.
Because selection uses the question, the passage is a *supporting* fact, not a
random document chunk; `oracle_passage` measures how often it suffices.

- Raw documents (thousands of tokens each) would overload any fixed-size `S`
  long before testing retrieval; that is a different (compression) experiment.
- Writing the question/answer pair itself would make "retrieval" trivial string
  matching and confound memory with memorisation of the question.
- With passages, the question wording is never written to memory; the model
  must map question → stored content → answer.

### D2 — Global memory with a capacity ladder

One memory `S` per **bank** of `N` passages from `N` distinct questions; all `N`
questions are then asked against that same `S`. Ladder
`N ∈ {1, 4, 16, 64, 256, 1024}` (upper end as compute allows).

- `N = 1` is the per-question (reading-comprehension) case;
- larger `N` is global factual memory where the other passages are distractors.

This answers "global vs per-question" empirically instead of by assumption, and
exposes the capacity limit (a `d_k × d_v` state stores at most ~`d_k` clean
associations; interference is expected to grow with `N`).

### D3 — Passages are written independently (no positional binding)

Each passage is encoded by its own forward pass starting at position 0, and keys
are taken **before RoPE**. Memory therefore stores content keys, not absolute
corpus positions (issue caveat "positional encoding / chunking").

### D4 — Update rule

The issue writes `S ← αS + βk⊗v`; strictly that is a *Hebbian / decayed linear
attention* update. The *delta* rule is `S ← αS + βk(v − kᵀαS)ᵀ`, which
overwrites the value stored under an existing key. Both are implemented
(`src/memwmlm/memory/delta_rule.py`); **delta is primary**, Hebbian is reported as
an ablation. Keys are L2-normalised.

## Variants

### Variant 1 — existing Qwen heads (no new projections)

For each layer `ℓ` and KV head `h`: keys `k = norm(k_norm(W_K x))` (pre-RoPE),
values `v = W_V x`, giving `S_{ℓ,h} ∈ R^{128×128}` (Qwen3-1.7B: 28 layers × 8
KV heads, head_dim 128 — the `d` sweep does not apply). At application time each
query head reads its group's `S` with `norm(q_norm(W_Q x))` and the read is mixed
into the softmax attention output before `o_proj`:
`o = o_softmax + g · read(S, q)`.

Stages, in order, each reported separately:

1. **V1-frozen:** no trained parameters; scalar `g` and write `β` chosen on a
   *calibration* split (train-split banks), never on the evaluation split.
2. **V1-gate:** learn per-layer/per-head gates only (~hundreds of parameters).
3. **V1-LoRA:** add application-side LoRA (records how much adaptation is needed).

Trained stages use TriviaQA **train** banks for interface training; evaluation
banks come from **dev** with disjoint questions and disjoint answer strings
(see Splits). This is a deviation from the issue's "general corpus" suggestion for
Variant 2 and is declared here for Variant 1 only.

### Variant 2 — trained writer/reader heads

New writer `W_K, W_V` and reader `W_Q` (width `d ∈ {512, 1024, 2048, 4096}`) on
final (or selected) Qwen hidden states; read injected through the existing gated
residual hook path (`FrozenQwenWithMemory` pattern). The interface is first
trained on a **non-TriviaQA** task (planned: key→value recall over clean text
passages, details to be fixed in an amendment before training), then evaluated
write-once on TriviaQA dev banks without TriviaQA training. A TriviaQA-train
fine-tuned interface is a separately labelled secondary condition.

## Data

- TriviaQA v1.0 RC archive, Wikipedia domain (`qa/wikipedia-{train,dev}.json`
  plus `evidence/wikipedia/`); URL, archive/file SHA-256 recorded in
  `triviaqa-rc-manifest.json`.
- Unfiltered `dev` (issue #5) is only used for cross-reference with the issue #5
  closed-book numbers.
- Banks are generated deterministically (fixed seed) and stored with a manifest
  (question ids, passage text, selection rule, hashes).

### Splits

- **Calibration/training banks:** from `wikipedia-train`.
- **Evaluation banks:** from `wikipedia-dev` only; questions whose normalised
  answer set intersects a training-bank answer are dropped from evaluation
  (novel-instance / answer-leakage control).
- Contamination stratification: every evaluation metric is also reported split by
  closed-book correctness of Qwen3-1.7B on that question.

## Conditions and controls (same banks, same prompt)

| Condition | Passages in context | Memory |
|---|---|---|
| `closed_book` | no | none |
| `oracle_passage` | the question's own passage only | none |
| `counterfactual_oracle` | its answer-swapped passage only | none |
| `full_context` | all `N` bank passages (while it fits) | none |
| `rag_bm25@k` (k = 1, 3) | top-k BM25 passages from the bank | none |
| `memory` | no | `S` of the bank |
| `null_memory` | no | `S = 0` with identical read path |
| `shuffled_memory` | no | `S` of `N` calibration (train) passages, i.e. same size, none of the bank's facts |
| `counterfactual_memory` | no | `S` of the bank with each passage's answer span replaced by another eval question's answer (fixed derangement; answer type is *not* matched) |

The counterfactual control is the cleanest test of retrieval vs parametric
recall: a model reading memory should output the *swapped* answer. It is scored
against the swapped answer (follow rate) and the original answer (resist rate).
`counterfactual_oracle` gives the in-context reference for the same swap.

Prompts: closed-book and all memory conditions use `triviaqa_short_answer_v1`
verbatim (issue #5). In-context conditions use
`triviaqa_short_answer_context_v1` (same instruction prefixed with "Use the
passages below if they are relevant." and numbered passages). Memory conditions
therefore receive no textual hint that memory exists.

## Metrics

- Primary: TriviaQA normalised exact match (aliases), greedy decoding,
  ≤ 32 new tokens, prompt `triviaqa_short_answer_v1` (identical to issue #5).
- Secondary: token F1; counterfactual follow rate; per-`N` curves.
- Paired bootstrap 95% CIs (10,000 resamples) for memory vs each baseline.
- Matched exposure: report passages written, tokens written, trainable
  parameters, optimizer steps, and inference cost per condition.

## Expected result / falsification

- **Supports C1:** memory EM ≫ closed-book and null at small `N`, within a few
  points of RAG/full-context, drops under shuffled memory and follows
  counterfactual swaps.
- **Refutes C1 (for a variant):** memory ≈ null/closed-book at all `N`; or gains
  do not follow counterfactual swaps (memory acts as a generic prompt/bias); or
  gains exist only on questions Qwen already answers closed-book.
- **Expected partial outcome:** V1-frozen near null (softmax keys are not built
  for linear readout); accuracy falling with `N` as the state saturates.

## Planned seeds

Bank construction seed `20261111`; training seeds `{0, 1, 2}` for trained stages.

## Artifacts

- Code: `src/memwmlm/memory/delta_rule.py`, this directory.
- Data: `artifacts/triviaqa/` (ignored), bank manifests here.
- Runs: `runs/delta-rule/` (ignored); summaries in `RESULTS.md` here.

## Amendments

Record dated protocol changes here before inspecting affected results.

- 2026-11-11 (before any eval memory run): diagnostics on calibration data
  (`NOTES.md`) showed that writing every passage token into a 128×128 head state
  retains passage-specific content only for N ≲ 16 and that frozen reads help
  only at N ∈ {1, 2}. Therefore:
  - **V1-frozen eval configuration (fixed from calibration questions 384–895
    only):** delta rule, β = 1, α = 1, all 28 layers × 16 query heads,
    scalar gate 0.2, key centring with the mean key of 256 calibration passages,
    first token of each passage not written.
  - Eval ladder for V1-frozen memory conditions: N ∈ {1, 2, 4, 16}. N = 2 is
    added (consecutive chunks of `eval.jsonl`, same rule as `banks.json`); larger
    N are not run because calibration and diagnostics show collapse by N = 4.
  - A Hebbian run at the same configuration is reported as an untuned ablation.
  - A question-blind robustness bank (`--selection first`, same eval question
    ids) is added to check the BM25/selection confound.
- 2026-11-12 (before any run with β ≠ 1 or α ≠ 1): the V1-frozen eval used
  β = α = 1 without a sweep. β = 1 fully overwrites the state along each new
  key, which with overlapping keys erases earlier tokens; and β = 1 Hebbian
  makes the state grow with tokens written, so the Hebbian ablation above is
  not a fair comparison. A **write-parameter sweep** is added, calibration
  split only, questions 384–895 (the slice used for the gate refinement), banks
  `banks-v2`, centred keys, all heads, ladder N ∈ {1, 2, 4, 16}, conditions
  `memory` and `shuffled_memory`:
  - Stage A (delta, α = 1): β ∈ {0.1, 0.3, 0.5, 1} × gate ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.2}.
  - Stage B (delta): α ∈ {0.999, 0.99, 0.95} at the Stage-A β, gates = the
    Stage-A best gate and its two neighbours in the grid.
  - Stage C (Hebbian, α = 1, β normalised by tokens written, `--beta-norm
    tokens`): gate ∈ {0.5, 1, 2, 4, 8, 16}.
  - **Selection criterion (fixed now):** score = mean over N ∈ {1, 2, 4, 16}
    of (memory EM − shuffled EM); ties (within 0.5 points) broken by mean
    memory EM. One delta configuration and one Hebbian configuration are
    selected and then run once on eval with all memory controls
    (`memory`, `null_memory`, `shuffled_memory`, `counterfactual_memory`).
    The 2026-11-11 eval run stays reported as the pre-registered result; the
    new run is reported as a post-hoc-tuned V1-frozen configuration.
- 2026-11-13 (before any global-memory run): **global memory is the primary
  setting from here on**, as in the issue ("read the corpus once; the
  application asks a question and reads `S`"). The bank ladder N ≤ 16 stays
  reported as a diagnostic. Design:
  - **Corpus** (`prepare_global.py`, output `runs/delta-rule/global-v1/`):
    all TriviaQA-train passages of the 1,024 calibration questions plus
    3,072 further train questions drawn with the same seed and rule (those
    whose aliases intersect any eval alias are dropped), plus the passages of
    the **even-indexed** half of the 1,024 eval (dev) questions. The
    odd-indexed half is never written. Passages are written once, in a fixed
    shuffled order (seed 20261113), into one state `S` per layer/KV head.
  - **Question groups:** `train_seen` (512 train questions used for
    training, passage in S); `train_heldout` (512 train questions never
    trained on, passage in S; used for validation/early stopping only);
    `dev_in` (512 dev, passage in S, never trained on) — **primary**;
    `dev_out` (512 dev, passage not in S) — novel question *and* novel fact,
    measures harm/generic effects.
  - **Write settings** (no forgetting for a global memory, α = 1): delta
    β = 0.5 (best α = 1 delta in the 2026-11-12 sweep) and Hebbian β = 1/T.
    Key centring as before. Writing always uses frozen Qwen (LoRA off).
  - **Stages:** V1-frozen-global (scalar gate, from the 2026-11-12 sweep:
    delta 0.3, Hebbian 2 and 4); V1-gate (per-layer × per-query-head gates,
    448 parameters, initialised at the frozen gate); V1-LoRA (gates + rank-8
    LoRA on `q_proj` and `o_proj` of every layer, application side only).
    Loss: cross-entropy on the answer surface form (the matched span of the
    passage) followed by `<|im_end|>`, closed-book prompt, read from the global
    `S`. Training questions = all train questions except `train_heldout`;
    `train_seen` is a fixed 512 subset of them. AdamW, batch 16, 2 epochs,
    lr 1e-2 (gates) and 1e-4 (LoRA); nothing is tuned on dev.
    Seeds {0, 1, 2}.
  - **Evaluation states for every trained model:** `S` (memory), `S = 0`
    (null), `S` with the `dev_in` passages replaced by their counterfactual
    versions (counterfactual; follow rate on `dev_in`).
  - **Fine-tuning control:** V1-LoRA trained identically with `S = 0`
    (measures what LoRA on TriviaQA-train alone gives).
  - References: closed-book; BM25@1 over the whole written corpus.
  - **Primary comparison:** `dev_in` EM of memory vs null with the same
    trained parameters, and vs the null-trained LoRA control; `dev_in` vs
    `dev_out` gain; counterfactual follow on `dev_in`.
  - **Prior expectation (stated before running):** diagnostic 2 showed that a
    128×128 head state written with ≥ 64 passages reads written keys no better
    than unwritten ones; the global corpus is ~4.6k passages. Near-null
    `dev_in` gain is expected for every stage; training can only exploit
    information that is still in `S`.
- 2026-11-13 (before any Variant 2 training): **Variant 2 protocol fixed.**
  - **Interface training data (general text, no TriviaQA):** the issue's
    "clean repro text" is read as this repo's WikiText-103 reproduction
    corpus. Task: sentence cloze (`prepare_wikitext_cloze.py`, output
    `runs/delta-rule/wikitext-cloze-v1/`): 120k train sentences (6–64 words),
    each with one answer-like span (1–4 capitalised words not
    sentence-initial, or a 3–4 digit number); the query is a 6–16-word window
    around the blank, so it never contains the full stored sentence.
    Answer-leakage control: units are dropped if the span is any TriviaQA
    alias of the global corpus questions or if the sentence contains any
    multi-word TriviaQA alias (counts in the manifest). Validation: WikiText
    validation split.
  - **Architecture** (`src/memwmlm/models/delta_interface.py`): frozen
    Qwen3-1.7B; writer and reader on the output of decoder layer 14 (of 28):
    `k = norm(W_K rms(h))`, `v = W_V rms(h)`, per-token
    `β = sigmoid(w_β·rms(h) + b)`, delta rule with α = 1 into one `d × d`
    state (exact chunk-parallel form, `delta_rule.write_chunked`); read
    `q = norm(W_Q rms(h))`, layer output `h + W_O (qᵀS)`; `W_O` zero-initialised.
    Passages are encoded one per sequence from position 0 (first token not
    written), written in random/fixed order into one `S`.
  - **Training:** one episode per step: N ∈ {512, 1024, 2048} random train
    sentences written, 32 of them queried with the closed-book prompt
    (`triviaqa_short_answer_v1`, question = "Fill in the blank: " + cloze),
    loss = cross-entropy on span + `<|im_end|>`. AdamW lr 3e-4, 100 warmup
    steps, cosine decay, weight decay 0.01, grad clip 1, 2,000 steps (64k
    queries) for every width. **d ∈ {512, 1024, 2048, 4096}**, seeds
    {0, 1, 2}; seed 0 of every width first.
  - **Evaluation** (`eval_v2.py`, final checkpoint, nothing tuned on
    TriviaQA): the global corpus (4,342 passages) written once; all four
    question groups (every TriviaQA question is novel for Variant 2);
    counterfactual S (dev_in passages swapped); corpus-size curve for dev_in
    (512 = dev_in only, 1,024, 2,048, 4,342); in-domain positive control:
    WikiText validation cloze at N = 1,024 and 4,342. `S = 0` equals
    closed-book exactly. Primary: dev_in memory vs closed-book (18.2) and vs
    BM25@1 (52.3), counterfactual follow; dev_out shows harm.
  - **Secondary (separately labelled, later):** the same interface
    fine-tuned on TriviaQA-train questions (train_other) against the global S.
  - **Prior expectation:** the in-domain cloze should work at N ≈ training
    sizes for large d; transfer to paraphrased TriviaQA questions is the open
    question; dev_in gain, if any, should grow with d.
- 2026-11-13 (after V2 seed 0; before any V2b training; nothing inspected on
  TriviaQA beyond the V2 seed-0 eval in NOTES): **Variant 2b — make the
  positive control pass first.** V2 seed 0 learned no retrieval (real S =
  S of other sentences in loss at every N, `probe_v2.py`); its TriviaQA eval
  stays recorded as the pre-registered V2 result.
  - **Architecture change:** next-token association. Within each passage,
    key from token t and value from token t+1 (`k_t = norm(W_K rms(h_t))`,
    `v_t = W_V rms(h_{t+1})`, β from h_t); reader unchanged.
  - **Training change:** N ∈ {32, 128, 512, 2048}; everything else as V2.
  - **Gate before TriviaQA:** development uses only WikiText validation.
    Every validation logs real-vs-wrong-S loss at N = 64 and 1,024. A V2b
    width proceeds to `eval_v2.py` only if, at the end of training, the real
    S has lower loss than wrong S at N = 1,024 and in-domain cloze EM at
    N = 1,024 exceeds closed-book. Changes made during development (on
    WikiText only) are logged here before the TriviaQA eval.
- 2026-11-16 (V2b development so far is logged in NOTES "V2b development
  pilots"; still nothing V2b evaluated on TriviaQA): **V2b development
  options and decision rule.** Options added to `train_v2.py`, all
  developed and judged on WikiText only:
  `--shift-values` (key token t → value token t+1), `--bias-state` (learned
  constant S0 added to every read), `--standardise` (fixed per-dimension
  standardisation of layer-14 states from 4,096 train sentences, replacing
  RMS norm; removes the massive-activation dim), `--full-cloze` (sanity
  check only: query = whole sentence with the span blanked), `--fewshot K`
  (K worked cloze examples, the last K train units, in every cloze prompt;
  the TriviaQA eval prompt stays the unchanged closed-book prompt).
  - **Next pilot:** `--fewshot 3 --standardise --shift-values`, cropped
    cloze (the real task), N ∈ {1, 4, 16}, lr 1e-3, 1,000 steps, d = 1024;
    judged by `probe_v2.py` (256 queries).
  - **Decision rule (fixed now):** V2b proceeds to the d sweep only if real-S
    loss is below wrong-S loss by ≥ 0.1 nats at N = 1 *and* real-S EM exceeds
    wrong-S EM at N = 64. Otherwise Variant 2 is reported negative; for
    completeness the pre-registered V2 is then rerun at d = 4096 (write chunk
    512 to fit memory) and evaluated, so all four widths are reported.
  - **If it passes:** the configuration is frozen, then trained with N ∈
    {16, 64, 256, 1024, 2048} (curriculum: small N first half), 2,000 steps,
    d ∈ {512, 1024, 2048, 4096}, seeds 0/1/2; each width passes the
    WikiText gate of the previous amendment before `eval_v2.py`. α: a decay
    check α ∈ {0.999, 0.99} at the best d (write-time, no retraining).
- 2026-11-16 (before any expanded-key run): **Variant 1 extension: expanded
  keys (capacity test).** Frozen Qwen heads as in global V1 (pre-RoPE,
  centred keys, every layer and head), but keys and queries pass through
  DPFP (Schlag et al. 2021; `delta_qwen.dpfp`): `x = [relu(k), relu(−k)]`,
  `φ(x) = concat_{j=1..ν} x ⊙ roll(x, j)`, then L2-normalised; state per KV
  head `D × 128` with `D = 256 ν`. ν ∈ {2, 4, 8, 16} ⇒ **D ∈ {512, 1024,
  2048, 4096}**, the issue's memory-size sweep. Nothing is trained (no LoRA).
  - Rules: delta β = 0.5, α = 1 (exact chunked write) and Hebbian β = 1/T.
  - Gate: selected per (rule, D) on `train_seen` memory EM from {0.1, 0.3,
    1, 3} (delta) and {2, 8, 32, 128} (Hebbian); ties → smaller gate. The
    selected gate's dev_in / dev_out / train_heldout / counterfactual are the
    result (all four groups are evaluated at every gate in one run and the
    selection is applied afterwards; dev numbers are not used to choose).
  - Expectation: if global V1 failed for capacity reasons (delta state
    ill-conditioned with 128-dim keys), larger D should restore a
    dev_in-specific gain (dev_in > dev_out, counterfactual follow > 0).
  - 2026-11-16 note: the first few-shot pilot was stopped at step ~200
    because examples packed into one user message made Qwen copy the example
    answers (no-memory loss 12 vs 6.7). `--fewshot` now renders the K
    examples as prior user/assistant chat turns (`ClozeFormatter`; K = 0 is
    byte-identical to the old prompt). With K = 3 the frozen model answers
    tersely; no-memory loss 6.72 → 6.47 on 64 val queries. Pilot and decision
    rule unchanged.
- 2026-11-16 (before any 4B result): **Optional check: Qwen3-4B.** Same
  protocol, data, prompt and scoring as 1.7B, with `Qwen/Qwen3-4B` at
  `1cfa9a7208912126459214e8b04321603b3df60c` (the issue #5 revision; 36
  layers, 8 KV heads, head dim 128, hidden 2,560). Outputs under
  `runs/delta-rule/qwen3-4b/`, queue `run_4b.sh`. Planned runs:
  1. **Global references:** closed-book and BM25@1 on all four groups.
  2. **Global V1 frozen:** delta β 0.5 at gates {0.1, 0.3, 1} and Hebbian
     β 1/T at gates {1, 2, 4, 8}; the gate is selected on `train_seen`
     memory EM (ties → smaller), as for expanded keys. 1.7B's calibrated
     gates are not carried over.
  3. **Global V1 adapted, seeds 0/1/2:** gates and LoRA r8 for delta (init
     gate 0.3) and Hebbian (init gate 2), plus LoRA trained with `S = 0`
     (`run_v1_global.sh` recipe unchanged).
  4. **Small banks (banks-v2, 1,024 dev questions), frozen, pre-registered
     delta β = α = 1, gate 0.2, centred keys:** N ∈ {1, 4, 16}; conditions
     closed_book, oracle_passage, counterfactual_oracle, memory,
     shuffled_memory, counterfactual_memory. This is where 1.7B had its only
     content signal (N = 1); the question is whether 4B extends it.
  5. **V2 pre-registered recipe at d = 1024, seed 0,** layer 18 (same
     relative depth as 14/28), hidden cache `hidden-Qwen3-4B-L18`;
     `eval_v2.py` on TriviaQA and `probe_v2.py` (right vs wrong memory).
  6. **V2b final configuration** (`--fewshot 3 --standardise
     --shift-values`, N ∈ {1, 4, 16}, lr 1e-3, 1,000 steps, d = 1024) with
     the same decision rule as 1.7B (loss gap ≥ 0.1 at N = 1 and real EM >
     wrong EM at N = 64). Evaluated on TriviaQA only if it passes.
  - **Criterion (fixed now):** 4B changes the answer of this issue only if a
    memory condition beats on dev_in both closed-book (or its own `S = 0`
    for adapted runs) and its answer-swapped memory, with paired 95% CIs
    excluding 0; or, on small banks, memory beats both shuffled and
    counterfactual memory at N ≥ 4. Otherwise the 1.7B conclusion is
    reported as holding at 4B.
  - **Code changes for 4B (no effect on 1.7B results):** the global V1
    state cache key now includes model and revision for non-1.7B models
    (1.7B keys and cached states unchanged); the V2 hidden cache is
    model-tagged and checked against its manifest; the V1 adaptation loss
    applies `lm_head` only at answer positions (mathematically the same
    loss; full logits exceeded 24 GB at 4B).
  - 2026-11-16 note (after the frozen 4B runs, before any adapted 4B
    result): the first 4B gate run ran out of GPU memory in training. 4B
    adapted runs use `--micro-batch 8`: each batch of 16 is processed in
    two halves with token-weighted gradient accumulation (exactly the
    whole-batch loss; verified identical with one micro-batch of 16; with
    micro-batches of 4 the 1.7B gates differ by ≤ 0.02 after 6 steps, from bf16 rounding
    at different padded shapes). 1.7B runs are unaffected (default = whole
    batch).
  - 2026-11-16 note (steps 1–5 finished; before any V2b 4B result): the
    4B V2b run (3 worked examples per prompt) ran out of GPU memory at its
    first step. It is run with `train_v2.py --micro-batch 8`: the 32 queries
    of a step in four token-weighted parts sharing one S (same gradient; on
    1.7B, 3 steps whole vs one part of 32 vs parts of 8 differ by ≤ 6e-5 in
    any weight, below one Adam step of 1e-3). Default unchanged (whole).
- 2026-11-17 (before any result of these runs): **4B coverage of the remaining
  1.7B sweeps**, at the user's request, queue `run_4b_sweeps.sh`, same 4B
  model/revision and protocol as the 4B amendment above:
  1. **V2 pre-registered recipe at d ∈ {512, 2048, 4096}, seed 0,** layer 18
     (d 4096 with `--chunk-size 512` in training and evaluation, as for
     1.7B); `eval_v2.py` on TriviaQA and `probe_v2.py --sizes 1,4,64,1024`
     for each. Outputs `runs/delta-rule/qwen3-4b/v2/d{d}-s0`.
  2. **V1 expanded keys, frozen,** ν ∈ {2, 4, 8, 16} (D = 512 … 4,096),
     delta β 0.5 at gates {0.1, 0.3, 1, 3} and Hebbian β 1/T at gates
     {2, 8, 32, 128} (the 1.7B grids); gate selected on `train_seen` memory
     EM, ties → smaller. Outputs `runs/delta-rule/qwen3-4b/global/expand/`.
  - **Criterion:** the same as the 4B amendment (a memory beats on dev_in
    both 4B closed-book and its answer-swapped memory, paired 95% CIs
    excluding 0). If a width/D hits a grid edge or runs out of memory, that
    is reported, not re-tuned.
  - 2026-11-17 note (d 512 and 2,048 finished; before any d 4,096 or
    expanded-key 4B result): the 4B V2 d 4,096 run ran out of GPU memory at
    its first training step, which stopped the queue. It is rerun with
    `train_v2.py --micro-batch 8` (token-weighted parts sharing one S; same
    gradient, as verified for the 4B V2b note above); the queue resumes
    with the expanded-key runs afterwards. Nothing else changed.
    Addendum (same day, still before any d 4,096 result): micro-batching
    alone also ran out of memory, in the write backward (S checkpoints at
    every chunk boundary of a 2,048-passage write, 64 MB each at d 4,096).
    Training therefore uses `--chunk-size 2048` (fewer boundaries; peak
    21.9 GB over 3 steps vs out of memory at chunk 256 and 512, 23.97 GB at
    1,024). The chunked write is exact for any chunk size
    (`tests/test_delta_rule.py`), and 3 steps at chunk 1,024 vs 2,048 gave
    the same weights (write-gate bias equal to 1e-9). Evaluation keeps
    `--chunk-size 512` as for 1.7B (no gradient there).
- 2026-11-18 (before any result of these runs): **Code corrections and a
  direct capacity test** (after an independent code review; no bug was found
  that makes memory content irrelevant, but three problems were):
  1. *Dropped first token.* Qwen3's tokenizer adds no BOS, so
     `skip_first_token` (meant to skip an attention sink) dropped the first
     content token of every written passage (in 61 of 1,024 banks-v2 eval
     passages the answer starts there). Fix: `prepend_sink` /
     `run_banks.py --sink-prefix` puts `<|endoftext|>` in front of each
     passage, so the skipped position is that token. All earlier runs wrote
     passages without their first token; they are not rerun (the effect is
     equal for real and answer-swapped memory).
  2. *Uncentred queries with expanded keys.* Keys were centred before DPFP,
     queries were not; DPFP is not shift-equivariant, so query–key
     similarity in the expanded space did not track the original one. Fix:
     `center_queries` / `--center-queries` subtracts the same key mean from
     queries. The earlier expanded-key runs (1.7B and 4B) used the
     mismatch.
  3. *Overstated conclusion.* "Capacity is not the bottleneck" does not
     follow: the global corpus writes ~184k tokens into states whose key
     dimension is at most 4,096 per head, so the expanded-key test never
     reached a regime where the corpus fits. Wording is corrected after this
     test, whatever its result.

  **Capacity test (small banks, banks-v2, frozen V1, delta β = α = 1,
  centred keys, `--sink-prefix`), Qwen3-1.7B then Qwen3-4B,** queue
  `run_capacity.sh`, outputs `runs/delta-rule/capacity/{1.7b,4b}/`:
  - Config **A**: plain keys (D = 128). Config **B**: DPFP ν = 16
    (D = 4,096) with `--center-queries`.
  - **Gate**, per config and model, chosen on the *calibration* split only
    (first 512 questions, N = 16): grid {0.05, 0.1, 0.2, 0.5, 1}; pick the
    gate with the largest memory − shuffled EM, ties → smaller gate.
  - **Eval** (1,024 eval questions): closed_book, memory, shuffled_memory,
    counterfactual_memory at N ∈ {1, 4, 16, 64}; paired bootstrap 95% CIs
    (10,000 resamples) as for the earlier small-bank results.
  - **Criterion (fixed now):** a config "has signal at N" if memory −
    shuffled and memory − counterfactual both have CIs above 0. Let N*(X)
    be the largest N with signal (0 if none). If N*(B) > N*(A), the
    Variant 1 read is capacity-limited (more key space extends it) and the
    write-up says so; if N*(B) ≤ N*(A), more key space does not extend the
    read, and the limit is the read itself. Descriptive only: A at N = 1
    vs the earlier run without the sink fix.
- 2026-11-18 (before any result of these runs): **Global expanded-key rerun
  with both fixes**, at the user's request. `train_v1_global.py` gains
  `--sink-prefix` and `--center-queries` (defaults unchanged, old cache keys
  unchanged). Runs: frozen V1, D = 4,096 (ν = 16), delta β 0.5 at gates
  {0.1, 0.3, 1, 3} and Hebbian β 1/T at gates {2, 8, 32, 128}, Qwen3-1.7B
  and Qwen3-4B, gate selected on `train_seen` (ties → smaller), exactly as
  the original expanded-key runs; queue `run_expand_fixed.sh`, outputs
  `runs/delta-rule/expand-fixed/{1.7b,4b}/`. Criterion as before: a run
  changes the answer only if dev_in memory beats both closed-book and the
  answer-swapped memory with paired 95% CIs above 0. These replace the
  flawed D 4,096 numbers as the reference; the others stay labelled.
- 2026-11-18 (before any result of these runs): **1.7B capacity-test
  sensitivity check**, at the user's request. The 1.7B config A gate was
  selected at the grid edge (0.05) on calibration differences within noise,
  which weakens the A-vs-B comparison. Labelled sensitivity check, not a
  replacement of the pre-registered result: 1.7B A at gates 0.1 and 0.2 and
  1.7B B at gate 0.2 (B at 0.1 is the existing `eval-B`), same eval split,
  N ∈ {1, 4, 16, 64}, memory / shuffled / counterfactual, same CIs and the
  same N* criterion, compared at equal gate (A vs B at 0.1, A vs B at 0.2).
  If N*(B) > N*(A) at either gate, the write-up says that the 1.7B result
  depends on the gate; otherwise the caveat is narrowed. Queue
  `run_capacity_sens.sh`, outputs `runs/delta-rule/capacity/1.7b/sens-{A,B}-g{gate}`.
