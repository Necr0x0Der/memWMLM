# Final control: continuation meaning or additional positions?

This experiment closes the current Direction C probe series by clarifying why
spatial relations become more decodable after a question, before generation.
Decoder transfer, a new dataset, and causal interventions are outside its scope.
Measured outcomes are recorded in [RESULTS.md](RESULTS.md).

## Conditions

For each `none/shuffled` situation, extract features under ten conditions:

| Condition | Continuation after the situation |
|---|---|
| no_continuation | None |
| Six pairwise questions | Each of the six `Is X left of Y?` questions in the manifest |
| general_spatial | `Consider everyone's relative positions.` |
| neutral_complete | `The passage is now complete.` |
| neutral_sentences | `This passage contains several sentences.` |

All six questions are retained to avoid selecting the best question from the
previous test. The leftmost question is excluded as a continuation because it
is a different query type. **All seven targets, including leftmost, are probed
under every condition.**

Append the continuation after two newlines without a `Question:` label. Use the
same chat template, `enable_thinking=False`, and assistant prefix throughout;
no answers, answer options, or generated reasoning are provided. Before
extraction, require exactly equal full-prompt token lengths across all nine
continuation conditions for every row in every split. A mismatch raises an
error. The no-continuation baseline is intentionally shorter. Wording is fixed
before the run and must not be tuned using test results.

A preliminary local check with the available Qwen3-1.7B tokenizer found exactly
six added tokens for every continuation on all 576 situations. This is only a
technical preliminary check: the extractor repeats the check using the actual
pinned 4B tokenizer. Neutral phrases are not assumed to be semantically empty.
Two variants expose some wording dependence but do not exhaust possible controls.

## Fixed protocol

- Frozen Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16.
- Existing `../artifacts/db` dataset: 408/72/96 rows and 17/3/4 independent
  abstract-layout groups in train/validation/test. Do not regenerate it.
- Fresh extraction for **all** conditions, including baseline and questions.
  Do not mix old and new features: the previous protocol used `Question:`.
- Base transformer only, without LM head or generation. Layers 8/16/28/36;
  last nonpadding assistant-prefix token and mean over all nonpadding tokens.
- One separate linear classifier per target and condition: 70 classifiers.
  Each selects pooling/layer/L2 on validation only. L2: 0.0001 and 0.001;
  LBFGS max_iter=100, seed=12012. Lock all selections before reading test
  features. Normalization uses training data only.
- Another 70 control classifiers use one group permutation of training labels
  at the selected configuration. This is a diagnostic control, not a sampled
  null distribution.
- Metrics: mean balanced accuracy across seven targets, per-target scores,
  and joint accuracy of all targets. The targets are logically dependent and
  do not describe a complete world state.
- Paired bootstrap intervals: 2,000 group draws, using the same sampled groups
  on both sides. Intervals condition on selected probes and omit multiplicity
  corrections and training/selection uncertainty.

## Primary comparisons and interpretation

1. `questions_vs_neutrals`: average score of the six questions minus average
   score of the two neutral phrases.
2. `general_vs_neutrals`: general spatial instruction minus average score of
   the two neutral phrases.

Also save both comparisons against each neutral phrase separately, the general
instruction against the question average, and every condition against baseline.
Average **metrics**, not predictions: this is not an ensemble. Questions and
paraphrases do not count as additional independent worlds.

- Questions outperform both neutral controls: evidence for a contribution from
  a directed spatial query beyond equal continuation length.
- The general instruction also helps: relations can be made more accessible
  without specifying a particular future question, under the tested readouts.
- Neutral continuations also help: some improvement does not require an explicit
  spatial task. This is compatible with additional processing but does not
  itself prove a “more computation” mechanism.
- Small or nonsignificant differences do not establish equivalence. Report
  disagreement between neutral controls rather than selecting a convenient
  control after inspecting test results.

Source text remains available during encoding; mean pooling includes the
continuation. No outcome establishes a complete prepared world state, its
absence, causal use of the features, or operation without source context.
Four previously inspected test layouts limit generalization. These limitations
belong in the conclusion; they do not require extending this experimental series.

Commands and output files: [RUNBOOK.md](RUNBOOK.md).
