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
- 2026-11-19 (before any eval result of these runs): **Upper bound for
  Variant 1: uncompressed key–value memory with a softmax read**, at the
  user's request after the sanity checks (`NOTES.md` 2026-11-19). Question:
  can Qwen's own queries find the right stored passage tokens when there is
  no compression and no blur? `DeltaMemoryConfig(read="knn")` /
  `run_banks.py --read knn`: every written token's key and value is kept per
  KV head (no delta rule); each query attends with one joint softmax over
  its normal context and its top-k stored keys. Stored keys are RoPE-rotated
  right-aligned (token j of a passage of L tokens at position j − L), so each
  passage sits immediately before the prompt, all passages sharing that slot
  (exact equivalence to the passage pasted in context for one passage and
  k = all is a unit test). Sink prefix on, sink excluded.
  - Development on the calibration split only: position-free stored keys
    (distance 0) swamp attention and were dropped. Setting chosen on
    calibration (first 256 questions, N = 16) by memory − shuffled EM, ties
    → fewer keys: layers {all, 7–27, 14–27, 21–27} × k {8, 32}, then k
    {1, 2, 4, 16} for all layers. Selected: **all layers, k = 4** (36.7 vs
    shuffled 9.0; closed-book 18.0).
  - Eval: banks-v2 eval split (1,024 questions), N ∈ {1, 4, 16, 64, 256,
    1024}; conditions closed_book, memory, shuffled_memory,
    counterfactual_memory; references from `eval-baselines` (oracle passage,
    BM25@1, full context). Paired bootstrap 95% CIs as before.
  - Criterion (fixed now): signal at N if memory − shuffled and memory −
    counterfactual both have CIs above 0; N* = largest such N. If N* ≥ 64
    (Variant 1 delta-rule N* was 0–1), Qwen's own query→key matching does
    find stored content and Variant 1's failure is due to compression/blur
    (so spreading keys, sharper reads and more room are worth pursuing); if
    N* ≤ 4, the frozen matching itself is the wall. Also reported: memory vs
    closed-book (CI), counterfactual follow rate, and the gap to BM25@1 and
    full context.
- 2026-11-19 (before any eval result of these runs): **Whitened keys for
  Variant 1** ("spreading the keys"), at the user's request. Diagnostic
  (calibration passages only, `NOTES.md`): centred V1 keys use ~17–18 of 128
  dimensions (effective rank); per-head whitening with calibration key
  statistics, `W = (C + ε·mean_eig·I)^(-1/2)` applied to centred keys and
  centred read queries (`--whiten`, ε = 0.1 fixed a priori), raises this to
  ~88 and improves written-vs-unwritten self-retrieval at N ≤ 16. Test with
  the capacity-test protocol (README amendment 2026-11-18), Qwen3-1.7B,
  banks-v2, frozen delta β = α = 1, centred keys, sink prefix: config **W** =
  whitened D 128, config **WB** = whitened + DPFP ν 16 (D 4,096); gate per
  config chosen on the calibration split (N 16, 512 questions, memory −
  shuffled, grid {0.05, 0.1, 0.2, 0.5, 1}, ties → smaller); eval N ∈ {1, 4,
  16, 64}, closed_book / memory / shuffled / counterfactual, paired CIs.
  Criterion (fixed now): whitening extends the content signal if N\*(W) >
  1 or N\*(WB) > 1 (1 = the best N\* of the unwhitened configs A and B at
  1.7B over the selected and equal-gate checks); otherwise it does not. Also
  reported: memory vs closed-book. Queue `run_whiten.sh`, outputs
  `runs/delta-rule/whiten/1.7b/`.
- 2026-11-19 (before any result of these runs): **Two more Variant 1 sanity
  checks**, at the user's request (Qwen3-1.7B, banks-v2, frozen delta β = α
  = 1, centred keys, sink prefix; configs A = D 128, B = DPFP ν 16 with
  centred queries, W / WB = whitened as above).
  1. **Self-read** (`sanity_selfread.py`): fill the memory exactly as in
     `run_banks.py` with N eval passages (N 1–1,024; small N averaged over
     banks up to 64 passages), then read it through the model's read path
     (gate 1) using the written tokens' *own keys* as queries — storage
     only, no question→passage matching. Per head: centred cosine(read,
     stored value), top-1 identification of the right token among all
     stored values (chance 1 / tokens), and the cosine for keys of unwritten
     passages (control). Descriptive; no pass/fail criterion. Reading: if
     top-1 is high at N where QA fails, storage works and the failure is the
     question-side matching; if low, the memory does not store tokens
     retrievably even by their own keys.
  2. **One question → one answer (QA-pair memory)**: new conditions in
     `run_banks.py`: `qa_memory` writes, instead of the passage, the exact
     closed-book prompt of the question followed by the gold answer (so the
     question's own tokens are the stored keys); `shuffled_qa_memory` = the
     same for N calibration questions; `counterfactual_qa_memory` = the same
     prompt with the donor's (answer-swapped) answer, scored for follow.
     Configs A and B; gate per config on the calibration split (512
     questions, N 1, `qa_memory` − `shuffled_qa_memory`, grid {0.05, 0.1,
     0.2, 0.5, 1}, ties → smaller, gates where both EMs are < 1% skipped as
     collapsed: `select_capacity_gate.py ... 1 qa_memory --skip-collapsed`).
     Eval N ∈ {1, 4, 16, 64}: closed_book, qa_memory, shuffled_qa_memory,
     counterfactual_qa_memory, plus passage `memory` at N 1 with the same
     gate for comparison. Criterion (fixed now): the QA-pair memory works at N
     if qa_memory − shuffled_qa and qa_memory − counterfactual_qa both have
     95% CIs above 0; it *helps* if also qa_memory − closed-book > 0 (CI).
     Also reported: follow rate of the counterfactual answer. Queue
     `run_sanity3.sh`, outputs `runs/delta-rule/sanity3/`.
- 2026-11-20 (before any full result; a 64-question smoke run was used only
  to test the code): **Sanity check 5 — read memory with the same passage**,
  at the user's request (check 1 as intended: does retrieving by the passage
  itself, not the question, help?). `sanity_cloze.py`, Qwen3-1.7B,
  banks-v2: memory filled as in `run_banks.py` (banks of N eval passages,
  sink prefix, delta β = α = 1); at test time Qwen gets no question but the
  passage cut right before the answer (sink + passage text up to the answer
  span, identical tokens to the written ones) and continues greedily (12
  tokens). Hit = normalised continuation starts with the gold answer (alias
  or surface). Items with < 3 prefix words skipped. Conditions: none
  (prefix only), memory, shuffled (N calibration passages), counterfactual
  (answer-swapped bank; follow rate), oracle (full passage in context
  before the prefix). Configs A (D 128), B (DPFP ν 16), WB (whitened + DPFP
  ν 16), gate chosen per config and N on the calibration split (512
  passages, memory − shuffled, grid {0.05 … 1}, ties → smaller, collapsed
  gates skipped); K = uncompressed softmax memory (top-k 4, upper bound, no
  gate). N ∈ {1, 4, 16, 64}. Criterion (fixed now): reading by the same
  passage gives a content signal at N if memory − shuffled and memory −
  counterfactual both have paired 95% CIs above 0, and helps if also
  memory − none > 0 (CI); N* per config. Outputs `runs/delta-rule/cloze/`.
- 2026-11-20 (before the full training run and any eval; a 100-step smoke
  run only tested the code): **Trained retrieval for the delta-rule memory
  ("Variant 1T")**, at the user's request, keeping the delta rule. Sanity
  checks 3–5 (`NOTES.md`) showed that the uncompressed softmax memory reads
  stored passages (cloze 98.6% at N 1, 64.6% at N 64) while the delta-rule
  read returns a blur even with exact queries. Method: per-head learned
  feature maps (Hedgehog-style, Zhang et al. 2024; `FeatureMaps` in
  `delta_qwen.py`): `phi(x) = [softmax(Wx+b), softmax(−(Wx+b))]`, W
  initialised to I, one map per (layer, KV head) for keys (L2-normalised)
  and per (layer, query head) for queries, plus a per-head read scale; 11.1M
  parameters, Qwen frozen. The delta rule (β = 1) writes `phi_k(k)` → v and
  is read with `phi_q(q)`, added to the attention output as in Variant 1.
  Training (`train_feature_map.py`): layer-wise distillation from the
  softmax memory (top-k 4) on TriviaQA *training* passages that are not in
  banks-v2 eval/calibration (2,550 train, 256 validation): target = what
  the softmax memory adds to each head's attention output on the bank's QA
  prompts (+ gold answer) and cloze prompts; loss = squared error / mean
  square of the target (1 = no better than reading nothing); N ∈ {1, 4, 16,
  64} uniformly, 3,000 steps, AdamW lr 1e-3 one-cycle, seed 0; checkpoint =
  best validation error. Evaluation, gate fixed at 1 (the scale is
  trained; no tuning on eval): (1) QA with the capacity-test protocol
  (banks-v2 eval, 1,024 questions, N ∈ {1, 4, 16, 64}, closed_book /
  memory / shuffled / counterfactual, paired CIs); (2) the same-passage
  cloze test (sanity check 5). Criterion (fixed now): trained retrieval
  works if on QA N\*(T) > 1 (best delta-rule N\* so far: 1), and helps if
  memory − closed-book has a CI above 0 at some N ≥ 4. Also reported: the
  validation distillation error by N and layer, and the cloze table next to
  configs A/B/WB/K.
- 2026-11-23 (before any run of it; only unit tests ran): **Sanity check
  6 — answer the question, retrieve with the passage**, at the user's
  request (check 1 exactly as intended; the cloze test of check 5 changed
  the prompt). Same protocol as the capacity test (`run_banks.py`, banks-v2,
  1,024 eval questions, the usual closed-book question prompt, N ∈ {1, 4,
  16, 64}), but in every layer the memory is read not with each prompt
  token's own query: when a passage is written, its pre-RoPE query is
  averaged over the written tokens (per layer and head); every token of the
  question (and every generated token) reads the memory with the averaged
  query of the question's own passage. The context attention is unchanged.
  Conditions (new in `run_banks.py`): `passage_query_memory` (bank state,
  own passage's query), `shuffled_passage_query_memory` (N calibration
  passages, own passage's query; facts absent),
  `counterfactual_passage_query_memory` (answer-swapped bank, query of the
  stored swapped passage; follow rate), plus closed_book. Configs A
  (centred, D 128), B (DPFP ν 16), WB (whitened + DPFP ν 16): gate chosen
  per config and N on the calibration split (first 512 questions, memory −
  shuffled of the new conditions, grid {0.05 … 1}, ties → smaller,
  collapsed gates skipped); K = uncompressed softmax memory (top-k 4, no
  gate; the averaged query is rotated to the reading token's position).
  Criterion (fixed now): retrieval by the passage gives a content signal at
  N if memory − shuffled and memory − counterfactual both have paired 95%
  CIs above 0 (N\* per config, as before), and helps if also memory −
  closed-book > 0 (CI). Compared with the question-query results of the
  same configs (N\* = 1 at best). Outputs `runs/delta-rule/passage-query/`.
- 2026-11-23 (after the check 6 results): **Sanity check 6, fixed-gate
  row.** The per-N gate choice made the check 6 scores incomparable across N
  (A: 9.4 / 17.1 / 4.9 at N 4 / 16 / 64 used gates 0.2 / 0.05 / 0.05). For a
  like-for-like comparison across N, the eval is rerun with the same gate
  0.05 (the smallest grid value) for every N ∈ {1, 4, 16, 64} and configs
  A, then B, WB (same conditions, `--tag <config>-fixed`). Descriptive only:
  the N\* criterion above is unchanged.
- 2026-11-24 (before any training or evaluation; only a smoke run to test
  the code): **V1-LoRA on small banks**, at the user's request. The
  original V1-LoRA (issue nuance "it likely needs LoRA / interface
  adaptation") was trained only with the global memory, whose delta state
  was ill-conditioned (Frobenius norm ~250× a small-bank state), so it never
  tested whether LoRA can teach Qwen to read a *small* memory. Method
  (`train_lora_banks.py`): config A (delta β = α = 1, centred keys, D 128,
  sink prefix); writing always with frozen Qwen (LoRA off); application
  side rank-8 LoRA on `q_proj` and `o_proj` of every layer (α 16, 1.84 M
  parameters, the original recipe) plus per-layer × per-query-head gates
  (448, init 0.1). Training data: TriviaQA *train* questions with passages
  that are not in banks-v2 eval/calibration (the Variant 1T pool, 2,806
  rows; 256 held out for validation loss). Each step: N ∈ {1, 4, 16}
  uniformly; 8 prompts from 8 / N banks of N fresh random passages; each
  prompt reads its own bank's state; loss = cross-entropy on the gold answer
  surface + `<|im_end|>` after the closed-book prompt. AdamW (LoRA lr 1e-4,
  gates 1e-2), 2,000 steps, seed 0, final checkpoint (no selection).
  **Control (same recipe, seed 0):** LoRA trained with *unrelated* memory
  (each prompt reads a bank of N other random training passages, its own
  passage absent): learns the same task and read perturbation but cannot
  learn to use memory. Evaluation (`run_banks.py --lora`, gates = trained
  values, no tuning): banks-v2 eval, 1,024 questions, N ∈ {1, 4, 16, 64},
  closed_book (LoRA on, no memory), memory, shuffled_memory,
  counterfactual_memory (follow rate), for both LoRAs; frozen closed-book
  17.1 as reference. Criterion (fixed now): N\* per LoRA as before (memory
  − shuffled and memory − counterfactual CIs > 0). **LoRA makes the memory
  work** if N\*(LoRA) > 1 (frozen A: 1) and at N\* memory(LoRA) −
  memory(control LoRA) has a CI > 0 (the gain is not plain fine-tuning);
  **it helps** if also memory − closed-book (same LoRA) has a CI > 0 there.
  Outputs `runs/delta-rule/lora-banks/`.
- 2026-11-24 (after the V1-LoRA small-bank result, before any run of
  this; only unit tests and a smoke run): **V1-LoRA + learned β, option
  1**, at the user's request (option 2, a per-token β from the written
  token's hidden state, only if option 1 does not help). Same recipe,
  data, seed and evaluation as "V1-LoRA on small banks", plus a trained
  write strength per layer × KV head, β = sigmoid(θ) (224 parameters, init
  0.95, lr 1e-2). The write uses frozen Qwen keys/values as before but is
  differentiable in β (exact chunked delta rule, `write_chunked`; unit test:
  identical to the fixed-β write). Control: the unrelated-memory LoRA with
  learned β (same recipe). Validation banks are drawn per fixed seed and
  rebuilt with the current β (not comparable to the previous run's
  validation numbers; the evaluation is). Criterion (fixed now): learned β
  **helps** if, at N = 16 or N = 64, memory beats shuffled and
  counterfactual (CIs > 0) **and** memory(β-LoRA) − memory(fixed-β LoRA,
  the previous run) has a CI > 0 at that N; the N\* and control-LoRA
  criteria of the previous amendment are reported as well. Outputs
  `runs/delta-rule/lora-banks/1.7b/*-beta*`.
- 2026-11-24 (after option 1's result: criterion not met; before any run
  of this, only unit tests and a 30-step smoke run): **V1-LoRA + per-token
  β, option 2**, per the user's plan. Same recipe, data, seed, control
  and evaluation as option 1, but β is computed for every written token:
  β = sigmoid(w_{l,h} · x_t + b_{l,h}), x_t = the token's attention input
  (post `input_layernorm`) at layer l from frozen Qwen, per layer × KV head
  (`TokenBeta`, 28 × 8 × 2,049 = 459 k parameters; w = 0 and b =
  logit(0.95) at init, i.e. option 1's start; lr 1e-4 — the smoke run at
  1e-3 saturated β within 15 steps). Criterion (fixed now): as option 1,
  compared with the fixed-β LoRA: helps if at N = 16 or 64 memory beats
  shuffled and counterfactual (CIs > 0) and memory(per-token β) −
  memory(fixed-β LoRA) has a CI > 0 at that N; also reported vs option 1.
- 2026-11-24 (after option 2's result: criterion met at N = 16, nothing
  at N = 64; before any run of this, only unit tests and a 50-step smoke
  run): **per-token β + bigger memory**, at the user's request, to push past
  16 passages. Same recipe, data, seed, control and evaluation as option 2
  (per-token β, LoRA rank 8 on q/o_proj, per-head gates, training sizes
  N ∈ {1, 4, 16}, 2,000 steps, final checkpoint), with one change: the
  memory is config B — keys and read queries centred with the calibration
  key mean, then DPFP ν = 16 (`--key-expand 16 --center-queries`; key width
  128 → 4,096, state 32× larger). The learned-β write with expanded keys is
  the same exact chunked delta rule, checkpointed per layer and per chunk
  (unit test: identical to the fixed-β expanded write at constant β, and
  the read with grad equals the read without). Criterion (fixed now): the
  bigger memory **pushes past 16** if at N = 64 memory beats shuffled and
  counterfactual (CIs > 0) **and** memory(this) − memory(option 2) has a
  CI > 0 at N = 64; the N\*, control-LoRA and closed-book comparisons of
  the earlier amendments, and the difference vs option 2 at every N, are
  reported as well. Outputs `runs/delta-rule/lora-banks/1.7b/*-dtbeta*`.
- 2026-11-24 (before any run of this): **sanity check — passages in the
  context instead of memory**, at the user's request. Frozen Qwen already
  has these (`runs/delta-rule/eval-baselines`, original stage: oracle
  passage, answer-swapped passage, full context of the N bank passages);
  `oracle_passage` is rerun once to confirm it reproduces. New: the same
  conditions with the option 2 LoRA on (per-token β run, memory off):
  closed_book, oracle_passage, counterfactual_oracle, full_context at N ∈
  {1, 4, 16, 64}. Reported next to the memory results of the LoRA runs,
  with paired bootstrap CIs; no pass/fail criterion (a reference, not a
  hypothesis). Outputs `runs/delta-rule/lora-banks/1.7b/context-tbeta`.
- 2026-11-25 (before any run of this; only a 64-question smoke run):
  **Variant 2 on small banks (step A)**, at the user's request. The
  existing Variant 2 interfaces (trained on WikiText only; nothing
  retrained) were only evaluated on the global corpus (4,342 passages).
  Here they get the Variant 1 small-bank protocol (`eval_v2_banks.py`):
  banks-v2 eval, 1,024 questions, N ∈ {1, 4, 16, 64}; closed_book (S off),
  memory, shuffled_memory (N calibration passages, same offsets as
  `run_banks.py`), counterfactual_memory (follow rate). Checkpoints: the
  pre-registered V2 d ∈ {512, 1024, 2048, 4096} (seed 0, trained on banks
  of 512–2,048 sentences) and the final V2b pilot (d 1024, trained on banks
  of 1/4/16, `v2b-pilot/d1024-fewshot`). Each is run twice: plain Qwen, and
  with the Variant 1 option 2 LoRA on the application side (q/o_proj only;
  passages still written by plain Qwen; the user's suggestion). The LoRA
  was trained on TriviaQA *train* questions with Variant 1 memory, so the
  "+ LoRA" rows are labelled as such and compared with the LoRA's own
  closed-book (20.2), not with plain Qwen; the V2 interface itself stays
  general-corpus trained, as issue #6 requires. Criterion (fixed now): a
  run retrieves at N if memory beats shuffled and counterfactual (paired
  bootstrap CIs > 0); it helps at N if also memory − closed-book (same
  model) has a CI > 0. Outputs `runs/delta-rule/v2-banks/`.
- 2026-11-25 (after step A: no Variant 2 interface retrieves on small
  banks; before any run of this, only unit tests, a 100-step smoke run and
  a 64-question smoke evaluation): **Variant 2, step B — writer/reader
  initialised from Qwen's heads, trained on general text only**, at the
  user's request (issue #6: the interface is trained on a general corpus,
  not on TriviaQA). Every layer, every KV head (config A memory: centred
  keys, D 128, delta rule, sink prefix): writer = Qwen's k/v_proj (+ k_norm)
  plus a trainable rank-8 update (`WriterLoRA`; applied to the captured
  attention inputs, so Qwen's own pass over the passage is unchanged);
  reader = rank-8 LoRA on q_proj and o_proj (on only while answering);
  per-head gates (init 0.1); per-token β (`TokenBeta`, init 0.95, lr 1e-4).
  Untrained (both B = 0) it is exactly frozen Variant 1. Training
  (`train_lora_banks.py --task cloze --writer-rank 8 --learn-beta token`):
  WikiText-103 sentence cloze (`wikitext-cloze-v1`, already filtered
  against TriviaQA aliases), banks of N ∈ {1, 4, 16} sentences, 8 cloze
  prompts per step (V2b prompt with 3 worked examples), loss on the blanked
  word + `<|im_end|>`, 2,000 steps, seed 0, final checkpoint; key mean
  from 256 WikiText sentences. No TriviaQA question, passage or answer is
  used in training. Control: the same with unrelated sentences in memory.
  Evaluation (`run_banks.py --lora`; at evaluation the key mean is from the
  banks-v2 calibration split, as for every Variant 1 run): banks-v2 eval,
  1,024 questions, N ∈ {1, 4, 16, 64}, closed_book, memory,
  shuffled_memory, counterfactual_memory, for both models; plus the main
  model with the Variant 1 option 2 LoRA stacked (`--extra-lora`;
  labelled: that LoRA was trained on TriviaQA train, compared with its own
  closed-book only). Criterion (fixed now): **B retrieves** at N if memory
  beats shuffled and counterfactual (CIs > 0); **B works** if it retrieves
  at some N and there memory − control-model memory has a CI > 0; **it
  helps** if also memory − closed-book (same model) has a CI > 0. Also
  reported: WikiText validation loss with the right vs unrelated memory.
  The d ∈ {512…4096} sweep (bigger keys) only if B works. Outputs
  `runs/delta-rule/v2q/`.
