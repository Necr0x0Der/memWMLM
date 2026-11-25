# Qwen3-4B: instruction-specific probe results

Recorded 2026-11-25. This follow-up addresses instruction-specific classifiers
and the selective Mary test requested in [PR #14's review note](https://github.com/Necr0x0Der/memWMLM/pull/14#issuecomment-7113323224).

**Canonical targets are highly linearly decodable; shuffled targets remain much
less decodable even with separately trained classifiers. Instructions improve
some shuffled scores, but there is no clear selective improvement for Mary.**
The result does not establish a stable or causally usable world state.

## Run and provenance

- Source report: `artifacts/probes-qwen3-4b/report.json`.
- Predictions: `artifacts/probes-qwen3-4b/test_predictions.jsonl`.
- 56 checkpoints: seven questions × four instructions × two fact orders.
- Frozen `Qwen/Qwen3-4B`, requested revision
  `1cfa9a7208912126459214e8b04321603b3df60c`; 36 blocks, width 2560, BF16.
  The report records `resolved_revision: null`; the requested revision is pinned.
- Situation-only encoding. Candidates: layers 4, 8, 12, 16, 20, 24, 28, 36;
  `last_token`, `mean_pool`, `sentence_end_mean`; L2 = 0, 1e-5, 1e-4, 1e-3.
- Each question in each instruction/order cell independently selects pooling,
  layer and L2 on its own validation rows. Standardization and weights use train
  only; no train+validation refit. LBFGS budget: 100 iterations.
- Paired test-group bootstrap: 2,000 draws, seed 12012. Intervals are conditional
  on fitted/selected probes and are not corrected for multiple comparisons.
- Original dataset retained: 17/3/4 abstract-layout groups in train/validation/test;
  408/72/96 rows per instruction/order cell. Matched variants stay in one split.

The dataset-manifest hash matches the report. All test metrics were recomputed
from the 768 unique saved predictions and agree exactly; all 56 checkpoint
files are present. The feature directory recorded in the report is unavailable
in the current workspace, so feature provenance and checkpoint inference could
not be independently reverified here.

Report SHA256: `d8bab64d19f9aeac668da87bc716818146a199ef42c8a3368106673dfc53c7a3`.

Dataset manifest SHA256: `5e0c428da100d628a92c3414f055ff2ca2c8d51018f2c0b546b8a5843f25693c`.

Recorded feature manifest SHA256: `b96900c373d26edaed3b955594f0a73853a1c2585539128d24aa00b031d24dec`.

## Main metrics

Macro means mean per-question balanced accuracy (mean class recall). Joint
means all seven targets correct, not reconstruction of the full world state.
All values below are percentages. Each cell contains 96 test rows.

| Instruction | Order | Validation macro | Test macro | Test joint | Shuffled-label test macro |
|---|---|---:|---:|---:|---:|
| none | canonical | 100.00 | 96.43 | 87.50 | 57.24 |
| none | shuffled | 59.92 | 48.96 | 7.29 | 44.15 |
| imagine | canonical | 100.00 | 92.46 | 75.00 | 56.10 |
| imagine | shuffled | 58.63 | 56.50 | 10.42 | 52.28 |
| track_people | canonical | 100.00 | 95.63 | 83.33 | 55.80 |
| track_people | shuffled | 57.34 | 52.68 | 9.38 | 45.44 |
| track_mary | canonical | 100.00 | 95.14 | 82.29 | 55.26 |
| track_mary | shuffled | 62.80 | 54.96 | 13.54 | 49.70 |

Shuffled-label controls use one train-group permutation at the selected real-label
configuration. They are sanity checks, not optimized null baselines or permutation
significance tests. The nominal chance mean balanced accuracy for six binary
questions and one three-class question is 47.62%; the empirical null can differ
substantially with only four test groups.

## Paired instruction effects

Differences from `none`, in percentage points, separately within each fact order.
The same sampled groups are used on both sides of each contrast.

| Instruction | Order | Test macro difference | Paired 95% interval |
|---|---|---:|---|
| imagine | canonical | -3.97 | [-7.46, -0.30] |
| imagine | shuffled | +7.54 | [+5.36, +10.42] |
| track_people | canonical | -0.79 | [-1.79, +0.00] |
| track_people | shuffled | +3.72 | [-1.53, +9.52] |
| track_mary | canonical | -1.29 | [-2.68, +0.00] |
| track_mary | shuffled | +6.00 | [+2.38, +11.01] |

On shuffled rows, `imagine` and `track_mary` improve the aggregate relative to
`none` in this exploratory run. On canonical rows, instructions do not improve
the aggregate. These are differences between separately optimized linear
readouts, not a direct measure of how much the hidden representation changed.

## Selective Mary effect

Mary family: both directions of Mary–Bob and Mary–John (four probes). Control
family: both directions of Bob–John (two probes). `leftmost_person` is reported
separately because it involves all three people. Reverse directions are
logically redundant, not independent replications. Mary-family relations also
involve another person; this is relational selectivity, not isolated Mary rank.

`track_mary` minus `none`, in percentage points:

| Order | Quantity | Difference | Paired 95% interval |
|---|---|---:|---|
| canonical | mary_relations | +0.00 | [+0.00, +0.00] |
| canonical | bob_john_relations | -2.08 | [-4.17, +0.00] |
| canonical | selectivity_difference_in_differences | +2.08 | [+0.00, +4.17] |
| canonical | probe/leftmost_person | -4.86 | [-10.42, +0.00] |
| shuffled | mary_relations | +7.03 | [+0.00, +14.06] |
| shuffled | bob_john_relations | +7.29 | [+2.08, +12.50] |
| shuffled | selectivity_difference_in_differences | -0.26 | [-11.46, +10.94] |
| shuffled | probe/leftmost_person | -0.69 | [-4.17, +6.25] |

On shuffled rows, Mary relations improve by 7.03 points and Bob–John relations
by 7.29 points. Their difference is −0.26 points, with interval [−11.46, +10.94]:
**no clear selective Mary advantage**. This is not evidence of equivalence.
On canonical rows, Mary's score is unchanged near ceiling; the positive
interaction comes from a small Bob–John decline, not a Mary improvement.
Zero-width observed intervals do not establish population-level invariance.

## Pooling and layer selection

The test scores above combine independently selected representations; they
cannot be assigned to a single pooling method. Test results for all unselected
pooling candidates were not saved. The following comparison is **validation
only**: fix pooling, independently maximize validation balanced accuracy over
layer/L2 for each question, then average the seven scores. It is a selection
statistic, not a held-out estimate of pooling performance.

| Instruction | Order | last_token | mean_pool | sentence_end_mean |
|---|---|---:|---:|---:|
| none | canonical | 97.62 | 100.00 | 100.00 |
| none | shuffled | 54.17 | 57.24 | 58.73 |
| imagine | canonical | 99.40 | 100.00 | 100.00 |
| imagine | shuffled | 56.70 | 55.36 | 58.63 |
| track_people | canonical | 98.81 | 100.00 | 100.00 |
| track_people | shuffled | 55.95 | 54.27 | 54.96 |
| track_mary | canonical | 99.70 | 100.00 | 100.00 |
| track_mary | shuffled | 60.62 | 60.86 | 59.23 |

Canonical validation is saturated. For `none/canonical`, 92 candidates achieve
100% validation for each Bob–John direction; 15 for each Mary–Bob direction;
46 for each Mary–John direction; and 65 for leftmost. Ties prefer ordinary
accuracy, stronger L2, then first candidate. Selected early layers or pooling
frequencies therefore do not identify a unique state-storage location.

Selected configurations and individual test balanced accuracy:

| Condition | Question | Pooling | Layer | L2 | Test balanced accuracy |
|---|---|---|---:|---:|---:|
| imagine/canonical | bob_left_of_john | last_token | 4 | 0.001 | 84.38 |
| imagine/canonical | bob_left_of_mary | mean_pool | 8 | 0.001 | 100.00 |
| imagine/canonical | john_left_of_bob | last_token | 4 | 0.001 | 84.38 |
| imagine/canonical | john_left_of_mary | last_token | 8 | 0.001 | 91.67 |
| imagine/canonical | leftmost_person | last_token | 8 | 0.001 | 95.14 |
| imagine/canonical | mary_left_of_bob | mean_pool | 8 | 0.001 | 100.00 |
| imagine/canonical | mary_left_of_john | last_token | 8 | 0.001 | 91.67 |
| imagine/shuffled | bob_left_of_john | sentence_end_mean | 28 | 0.001 | 59.38 |
| imagine/shuffled | bob_left_of_mary | sentence_end_mean | 24 | 0.001 | 62.50 |
| imagine/shuffled | john_left_of_bob | sentence_end_mean | 28 | 0.001 | 59.38 |
| imagine/shuffled | john_left_of_mary | sentence_end_mean | 4 | 0.0001 | 61.46 |
| imagine/shuffled | leftmost_person | mean_pool | 12 | 0 | 29.86 |
| imagine/shuffled | mary_left_of_bob | sentence_end_mean | 24 | 0.001 | 62.50 |
| imagine/shuffled | mary_left_of_john | sentence_end_mean | 4 | 0.0001 | 60.42 |
| none/canonical | bob_left_of_john | last_token | 4 | 0.001 | 88.54 |
| none/canonical | bob_left_of_mary | mean_pool | 8 | 0.001 | 100.00 |
| none/canonical | john_left_of_bob | last_token | 4 | 0.001 | 88.54 |
| none/canonical | john_left_of_mary | last_token | 20 | 0.001 | 98.96 |
| none/canonical | leftmost_person | last_token | 16 | 0.001 | 100.00 |
| none/canonical | mary_left_of_bob | mean_pool | 8 | 0.001 | 100.00 |
| none/canonical | mary_left_of_john | last_token | 20 | 0.001 | 98.96 |
| none/shuffled | bob_left_of_john | sentence_end_mean | 8 | 0.0001 | 50.00 |
| none/shuffled | bob_left_of_mary | mean_pool | 16 | 0.0001 | 47.92 |
| none/shuffled | john_left_of_bob | sentence_end_mean | 8 | 0.0001 | 50.00 |
| none/shuffled | john_left_of_mary | sentence_end_mean | 36 | 1e-05 | 56.25 |
| none/shuffled | leftmost_person | sentence_end_mean | 16 | 1e-05 | 35.42 |
| none/shuffled | mary_left_of_bob | mean_pool | 16 | 0.0001 | 47.92 |
| none/shuffled | mary_left_of_john | sentence_end_mean | 36 | 1e-05 | 55.21 |
| track_mary/canonical | bob_left_of_john | last_token | 4 | 0.001 | 86.46 |
| track_mary/canonical | bob_left_of_mary | mean_pool | 8 | 0.001 | 100.00 |
| track_mary/canonical | john_left_of_bob | last_token | 4 | 0.001 | 86.46 |
| track_mary/canonical | john_left_of_mary | last_token | 12 | 0.001 | 98.96 |
| track_mary/canonical | leftmost_person | last_token | 8 | 0.001 | 95.14 |
| track_mary/canonical | mary_left_of_bob | mean_pool | 8 | 0.001 | 100.00 |
| track_mary/canonical | mary_left_of_john | last_token | 12 | 0.001 | 98.96 |
| track_mary/shuffled | bob_left_of_john | last_token | 4 | 0 | 57.29 |
| track_mary/shuffled | bob_left_of_mary | mean_pool | 36 | 1e-05 | 57.29 |
| track_mary/shuffled | john_left_of_bob | last_token | 4 | 0 | 57.29 |
| track_mary/shuffled | john_left_of_mary | mean_pool | 36 | 0.001 | 60.42 |
| track_mary/shuffled | leftmost_person | mean_pool | 36 | 0.001 | 34.72 |
| track_mary/shuffled | mary_left_of_bob | mean_pool | 36 | 1e-05 | 57.29 |
| track_mary/shuffled | mary_left_of_john | mean_pool | 36 | 0.001 | 60.42 |
| track_people/canonical | bob_left_of_john | last_token | 4 | 0.001 | 88.54 |
| track_people/canonical | bob_left_of_mary | mean_pool | 8 | 0.001 | 100.00 |
| track_people/canonical | john_left_of_bob | last_token | 4 | 0.001 | 88.54 |
| track_people/canonical | john_left_of_mary | last_token | 12 | 0.001 | 98.96 |
| track_people/canonical | leftmost_person | last_token | 8 | 0.001 | 94.44 |
| track_people/canonical | mary_left_of_bob | mean_pool | 8 | 0.001 | 100.00 |
| track_people/canonical | mary_left_of_john | last_token | 12 | 0.001 | 98.96 |
| track_people/shuffled | bob_left_of_john | mean_pool | 8 | 0.001 | 54.17 |
| track_people/shuffled | bob_left_of_mary | sentence_end_mean | 24 | 0 | 56.25 |
| track_people/shuffled | john_left_of_bob | mean_pool | 8 | 0.001 | 54.17 |
| track_people/shuffled | john_left_of_mary | last_token | 8 | 0.001 | 57.29 |
| track_people/shuffled | leftmost_person | sentence_end_mean | 12 | 0 | 33.33 |
| track_people/shuffled | mary_left_of_bob | sentence_end_mean | 24 | 0 | 56.25 |
| track_people/shuffled | mary_left_of_john | last_token | 8 | 0.001 | 57.29 |

## Full-context solvability versus decodability

The existing matched-budget Qwen3-4B full-context QA control reaches 93.07%
canonical / 95.00% shuffled test macro, and 64.58% / 73.96% joint-target accuracy,
aggregated across instructions. See [the parent results](../RESULTS.md) for
prompt, parsing and 2048-token generation-budget details. These aggregate QA
scores should not be mistaken for instruction-specific controls.

Unlike the earlier 1.7B results, this follow-up now measures situation-only
probes on the same model that has strong full-context QA. The shuffled probe
range (48.96–56.50%) is much lower than that model's full-context performance.
QA has the question and original trace and can generate intermediate reasoning;
these are different computational conditions. The gap does not identify the
mechanism or prove that a world state is absent.

## What remains unresolved

1. **Canonical surface shortcuts.** The canonical room chain is written left to
   right, followed by a fixed person order. Token positions and textual patterns
   can correlate with target relations. Splitting abstract layouts does not
   remove a format shared by train and test. High canonical scores may exploit
   this structure; lexical/one-hot baselines and controlled reorderings are
   still missing. This is a plausible explanation, not a demonstrated cause.
2. **Small effective sample size.** There are only three validation and four test
   layouts. Room-label permutations do not add independent layouts. Selection
   and bootstrap intervals are fragile; resamples may omit classes (recall is
   averaged over classes present). Intervals omit selection/training uncertainty.
   This benchmark was previously inspected, not a new confirmatory holdout.
3. **No causal-use test.** No source removal, matched null/shuffle/counterfactual
   state interface, or bottleneck continuation has been run. Shuffled training
   labels are not a causal intervention on the frozen model's state.
4. **Limited state schema.** Seven targets describe relative person order, not
   full room assignment, ownership, updates or temporal dynamics. This run also
   does not test new entity names or event compositions.
5. **Order sensitivity is not absence of state.** Separate shuffled training
   rules out only the explanation that the poor score comes entirely from
   applying a canonical-trained classifier. Pooling, linear readout capacity,
   data coverage and format dependence remain possible explanations. No
   cross-order transfer evaluation of these 4B banks was performed.

The two requested analyses are now implemented and measured. The defensible
claim remains condition-specific linear decodability, with strong dependence
on fact order and no clear Mary-selective benefit. The broader usable-world-state
claim remains open. Reproduction commands: [RUNBOOK.md](RUNBOOK.md).
