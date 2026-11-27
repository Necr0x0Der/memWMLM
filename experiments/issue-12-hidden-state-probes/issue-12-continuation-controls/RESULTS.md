# Continuation controls: measured results

**Specific pairwise questions improve spatial-relation decodability beyond the
two neutral continuations. The tested general spatial instruction does not
show a comparable benefit.** This closes the current Direction C probe series
with evidence for query-dependent accessibility, rather than proof of a complete
or causally usable world state.

Protocol: [README.md](README.md). Reproduction: [RUNBOOK.md](RUNBOOK.md).
Source artifacts: `artifacts/probes-qwen3-4b/`.

## Setup

- Frozen Qwen3-4B, requested revision
  `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, hidden width 2560.
- Existing `none/shuffled` situations: 408/72/96 train/validation/test rows,
  with 17/3/4 independent abstract-layout groups.
- Ten contexts: no continuation; six pairwise questions; the general instruction
  `Consider everyone's relative positions.`; and two neutral continuations,
  `The passage is now complete.` and `This passage contains several sentences.`
- Features precede any generated answer or reasoning. The protocol uses the
  base transformer, the same chat template and assistant prefix, no `Question:`
  label, and no supplied answers or answer options.
- Layers 8/16/28/36; last-token and mean pooling. Each of 70 linear classifiers
  independently selects pooling/layer/L2 on validation. L2 0.0001/0.001,
  LBFGS max_iter=100, seed=12012; all selections precede test access.
- Each selected classifier has a control fitted with one group permutation of
  training labels. These controls are not a null distribution or significance test.

The extractor requires exact equality of full-prompt token lengths across all
nine continuation conditions before extraction. However, the run's feature
manifest and raw features were not present in the reviewing workspace. Actual
4B prompt lengths and extraction provenance could therefore not be independently
reverified. The earlier +6-token check used the available 1.7B tokenizer only.

## Test results

Macro is mean balanced accuracy across all seven targets; joint means all seven
are correct in one situation. These are external probe predictions, not Qwen's
own answers. Each condition has 96 test situations. The seven targets describe
relative person order, not the complete world state.

| Continuation | Validation macro | Test macro | Joint | Shuffled-label macro |
|---|---:|---:|---:|---:|
| no_continuation | 54.22% | 47.62% | 9.38% | 47.52% |
| bob_left_of_john | 72.82% | 63.69% | 26.04% | 51.79% |
| bob_left_of_mary | 63.19% | 58.63% | 15.62% | 48.56% |
| john_left_of_bob | 68.65% | 61.21% | 15.62% | 48.81% |
| john_left_of_mary | 66.91% | 63.69% | 19.79% | 52.98% |
| mary_left_of_bob | 67.26% | 63.99% | 22.92% | 46.92% |
| mary_left_of_john | 73.71% | 59.52% | 15.62% | 52.68% |
| general_spatial | 56.94% | 49.21% | 8.33% | 47.32% |
| neutral_complete | 58.93% | 49.90% | 7.29% | 44.35% |
| neutral_sentences | 57.24% | 50.40% | 7.29% | 43.45% |

## Paired contrasts

Deltas in percentage points. Positive means the after condition is better.

| Contrast | Macro delta | Paired 95% interval |
|---|---:|---|
| bob_left_of_john_vs_no_continuation | +16.07 | [+6.55, +25.00] |
| bob_left_of_mary_vs_no_continuation | +11.01 | [-1.49, +23.51] |
| john_left_of_bob_vs_no_continuation | +13.59 | [+6.25, +20.54] |
| john_left_of_mary_vs_no_continuation | +16.07 | [+5.06, +26.49] |
| mary_left_of_bob_vs_no_continuation | +16.37 | [+2.98, +29.46] |
| mary_left_of_john_vs_no_continuation | +11.90 | [+2.98, +21.13] |
| general_spatial_vs_no_continuation | +1.59 | [-2.38, +5.56] |
| neutral_complete_vs_no_continuation | +2.28 | [-1.44, +5.70] |
| neutral_sentences_vs_no_continuation | +2.78 | [-3.27, +8.63] |
| questions_vs_neutrals | +11.64 | [+5.14, +18.07] |
| general_vs_neutrals | -0.94 | [-2.88, +0.40] |
| general_vs_questions | -12.58 | [-20.95, -4.75] |
| questions_vs_neutral_complete | +11.89 | [+4.15, +19.36] |
| general_vs_neutral_complete | -0.69 | [-1.59, -0.00] |
| questions_vs_neutral_sentences | +11.39 | [+6.13, +16.78] |
| general_vs_neutral_sentences | -1.19 | [-4.17, +1.49] |

## Interpretation

The six question conditions average **61.79% macro accuracy**, compared with
**50.15%** across the two neutral conditions and **47.62%** without continuation.
The primary question-minus-neutral contrast is **+11.64 percentage points**,
with a paired group-bootstrap 95% interval of **[+5.14, +18.07]**. It remains
positive against each neutral phrase separately: +11.89 and +11.39 points,
with both corresponding intervals excluding zero in this run.

Thus, under the protocol's length-matching requirement, merely appending text
of the same length is insufficient to explain the observed question advantage.
The content/form of a specific spatial question matters under these conditions.
This does not isolate spatial semantics from every other difference, such as
question syntax or the explicit naming of two people.

The general instruction scores **49.21%**, versus **49.90%** and **50.40%** for
the neutral phrases. Its primary contrast against their average is **−0.94
points [−2.88, +0.40]**. It provides no demonstrated advantage here and trails
the question average by **12.58 points**. This is a negative result for this
particular general instruction and readout procedure, not proof that all general
state-preparation instructions fail or that these conditions are equivalent.

Neutral continuations improve by only +2.28 and +2.78 points over baseline;
both intervals include zero. The general instruction's +1.59-point baseline
contrast also includes zero. These results do not establish a reliable benefit
from extra continuation positions alone.

For Direction C, the supported conclusion is that **the accessibility of spatial
relations in hidden features depends on the concrete query**. The result is
consistent with a question triggering integration of relevant information or
making existing information easier to read. It does not distinguish those
mechanisms, prove that no state existed before the question, or show that Qwen
uses the probed features causally. The simple attempt to prepare relations
through a general spatial instruction did not succeed in this run.

The current series can end with these conclusions. Its remaining limitations
are boundaries on the claim, not requirements to launch further experiments.

## Limits and comparison with earlier results

- Only four independent test layouts are available, and this benchmark has
  already been inspected. The 96 rendered rows per condition are not 96
  independent worlds.
- The 2,000-draw paired group-bootstrap intervals condition on fitted/selected
  classifiers, omit training/selection uncertainty and multiplicity correction,
  and can omit a class in a resample. Intervals reported here come from the
  saved report; they were not independently recomputed during review.
- Classifiers select representations independently across conditions. This is
  a comparison of decodability procedures, not a fixed-decoder intervention.
  Mean pooling includes the continuation itself.
- Both neutral phrases have meaning. Their agreement strengthens the particular
  contrast but does not establish that every neutral continuation behaves alike.
- Source tokens remain available during encoding. No source removal, state
  swap, or test of causal use in Qwen's own answers is included.
- The six-question aggregate concerns all seven targets. It must not be read
  as an estimate restricted to unasked relations; that requires a separate
  asked/inverse/other-target breakdown.
- Earlier question-conditioned results included a `Question:` label. These
  numbers use a fresh baseline and prompts without that label; they should
  not be treated as identical reruns of the earlier question conditions.

## Artifact verification and provenance

The review checked all 960 unique context/situation prediction records against
the dataset's target labels and group IDs. Per-target balanced accuracy, mean
macro accuracy, and joint accuracy were independently recomputed for all ten
real-label and shuffled-label conditions and agree with the report within
floating-point precision. All 140 classifier checkpoint files are present.
The local dataset manifest hash matches the report.

Feature extraction and checkpoint forward inference were not rerun. The feature
manifest is absent locally, so its recorded hash is provenance supplied by the
report, not an independently verified artifact hash.

Run repository commit recorded by the report: `52bf84aac5d699e27711a2d0a5c98a8b98963715`.

SHA256 values:

```text
report.json
  63e806f1c7eb75272f519609bf2d254e336e6a74310db7b5935eae35c6e48260
test_predictions.jsonl
  b21266e50b4df98388013386eceb17cd3ea19a99dc306873c6e7341684abea3b
dataset manifest (verified)
  5e0c428da100d628a92c3414f055ff2ca2c8d51018f2c0b546b8a5843f25693c
feature manifest (recorded; unavailable locally)
  9d70bf45f2b9aad98a4166568bc52b36e729633473fe319c947bb7e9bac8d041
```
