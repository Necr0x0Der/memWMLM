# Question-conditioned hidden states: measured results

Results inspected on 2026-11-27. Context: [issue #12](https://github.com/Necr0x0Der/memWMLM/issues/12).
Protocol: [README.md](README.md). Reproduction: [RUNBOOK.md](RUNBOOK.md).

**Adding a pairwise spatial question makes the target relations substantially
more linearly decodable before any answer tokens are generated. The improvement
extends beyond the asked relation and its inverse. The leftmost-person question
provides much weaker evidence of an improvement.**

These are external probe predictions, not Qwen's own answers. The result does
not establish a complete world state or demonstrate that Qwen causally uses
these features when answering.

## What was measured

- Frozen Qwen3-4B, revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16.
- The existing spatial dataset, `none/shuffled` condition: 408 train, 72
  validation, 96 test situations, grouped into 17/3/4 abstract layouts.
- Eight contexts per situation: no question, or one of the seven fixed questions.
  All use the same Qwen chat template, assistant prefix and `enable_thinking=False`.
  Neither ground-truth answers nor answer options are included.
- Extraction calls only the base transformer. No LM head, `generate()`, generated
  reasoning, memory adapter, or Qwen fine-tuning is involved.
- Layers 8, 16, 28, 36; last nonpadding token and mean over all nonpadding tokens.
  The last position belongs to the common assistant prefix, after the question
  when present. Mean pooling includes the question and chat tokens.
- Seven separate linear classifiers per context, 56 in total. Each independently
  selects pooling, layer and L2 on validation; L2 candidates are 0.0001/0.001,
  LBFGS max_iter=100. Normalization and weights use train only. All selections
  are locked before opening test features; there is no train+validation refit.
- Negative controls refit at the selected configuration after one fixed-seed
  permutation of complete target vectors across train groups. Seed: 12012.
- Paired group-bootstrap intervals use 2,000 draws of the four test layouts.

The no-question baseline was re-extracted with the same chat formatting.
Historical raw-text, no-question features are not this experiment's baseline.
The seven targets describe relative person order, not complete room assignments
or a full dynamic world state.

## Overall test results

Macro is mean per-question balanced accuracy. Joint counts situations where all
seven targets are correct. Every row has 96 test situations. Percentages below
refer to the whole seven-probe bank, not just the question supplied to Qwen.

| Input question | Validation macro | Test macro | All seven correct | Shuffled-label test macro |
|---|---:|---:|---:|---:|
| No question | 54.22% | 47.62% | 9/96 (9.38%) | 47.52% |
| Is Bob left of John? | 75.20% | 68.25% | 25/96 (26.04%) | 54.76% |
| Is Bob left of Mary? | 66.57% | 68.35% | 23/96 (23.96%) | 54.27% |
| Is John left of Bob? | 70.34% | 61.81% | 17/96 (17.71%) | 55.75% |
| Is John left of Mary? | 73.51% | 64.78% | 24/96 (25.00%) | 58.43% |
| Who is leftmost? | 56.15% | 51.09% | 6/96 (6.25%) | 47.82% |
| Is Mary left of Bob? | 68.25% | 70.83% | 26/96 (27.08%) | 51.39% |
| Is Mary left of John? | 74.90% | 62.30% | 14/96 (14.58%) | 51.88% |

For the six pairwise questions, overall test macro rises from 47.62% to
61.81–70.83%, a gain of 14.19–23.21 percentage points. All six corresponding
paired intervals for the overall macro difference exclude zero in this run.
This is an exploratory observation, conditional on the selected classifiers.
The leftmost question reaches only 51.09%; its overall difference interval
includes zero, and joint accuracy falls from 9/96 to 6/96.

## Asked target versus other targets

Compare each question-conditioned bank with the no-question bank on the same
test situations. For binary relations, exclude both the asked relation and its
inverse from the other-target family. For `leftmost_person`, the other family
contains all six directed relations. These families are not statistically or
logically independent: inverse pairs remain within the other family, and
transitivity and leftmost status link the targets.

Deltas and intervals are in percentage points.

| Question | Asked accuracy, before → after | Asked delta [95% interval] | Other targets delta [95% interval] |
|---|---:|---:|---:|
| Is Bob left of John? | 54.17% → 72.92% | 18.75 [6.25, 31.25] | 21.39 [10.83, 32.50] |
| Is Bob left of Mary? | 44.79% → 70.83% | 26.04 [14.58, 37.50] | 18.61 [6.25, 31.25] |
| Is John left of Bob? | 54.17% → 64.58% | 10.42 [-4.17, 25.00] | 15.69 [9.17, 22.08] |
| Is John left of Mary? | 54.17% → 75.00% | 20.83 [8.33, 33.33] | 15.69 [6.67, 26.67] |
| Who is leftmost? | 27.08% → 34.72% | 7.64 [2.08, 12.50] | 2.78 [-0.69, 6.25] |
| Is Mary left of Bob? | 44.79% → 77.08% | 32.29 [27.08, 37.50] | 19.58 [7.50, 32.64] |
| Is Mary left of John? | 54.17% → 65.62% | 11.46 [-12.66, 35.42] | 15.97 [3.75, 28.75] |

For example, after “Is Mary left of Bob?”, the asked relation improves from
44.79% to 77.08%. Other targets excluding that relation and its inverse improve
by 19.58 points on average. Bob–John improves from 54.17% to 71.88%, even though
it was not the question supplied to Qwen. This is broader than improvement on
one requested answer, but does not establish a question-independent world model.

The leftmost target itself rises from 27.08% to 34.72%. Its positive difference
from a weak baseline should not be mistaken for strong performance: three-class
chance balanced accuracy is 33.33%, and its matched shuffled-label control also
scores 34.72% on that target.

## Complete test matrix

Values are per-target balanced accuracy (%). Abbreviations denote directed
relations: B<J = Bob left of John, B<M = Bob left of Mary, J<B = John left of Bob,
J<M = John left of Mary, M<B = Mary left of Bob, M<J = Mary left of John.

| Input question | B<J | B<M | J<B | J<M | Leftmost | M<B | M<J |
|---|---:|---:|---:|---:|---:|---:|---:|
| No question | 54.17 | 44.79 | 54.17 | 54.17 | 27.08 | 44.79 | 54.17 |
| Is Bob left of John? | 72.92 | 67.71 | 72.92 | 70.83 | 54.86 | 67.71 | 70.83 |
| Is Bob left of Mary? | 67.71 | 70.83 | 67.71 | 66.67 | 68.06 | 70.83 | 66.67 |
| Is John left of Bob? | 64.58 | 65.62 | 64.58 | 56.25 | 59.72 | 65.62 | 56.25 |
| Is John left of Mary? | 61.46 | 68.75 | 61.46 | 75.00 | 43.06 | 68.75 | 75.00 |
| Who is leftmost? | 57.29 | 55.21 | 57.29 | 48.96 | 34.72 | 55.21 | 48.96 |
| Is Mary left of Bob? | 71.88 | 77.08 | 71.88 | 65.62 | 66.67 | 77.08 | 65.62 |
| Is Mary left of John? | 60.42 | 68.75 | 60.42 | 65.62 | 46.53 | 68.75 | 65.62 |

## Selected representations for the asked target

These settings describe only the classifier for the question that was supplied,
not the entire bank. Other targets independently selected their own settings.

| Input question | Pooling | Layer | L2 |
|---|---|---:|---:|
| Is Bob left of John? | last_token | 28 | 0.001 |
| Is Bob left of Mary? | mean_pool | 28 | 0.001 |
| Is John left of Bob? | mean_pool | 36 | 0.001 |
| Is John left of Mary? | last_token | 36 | 0.001 |
| Who is leftmost? | mean_pool | 8 | 0.001 |
| Is Mary left of Bob? | last_token | 36 | 0.001 |
| Is Mary left of John? | last_token | 28 | 0.001 |

Selections do not locate a unique world-state storage site. The final-layer
feature is after Qwen's final normalization, while the earlier features are
intermediate block outputs. Validation has only three independent layouts.

## What follows, and what does not

Supported by this run:

- After a pairwise question, linear readouts recover spatial relations more
  accurately than from the matched no-question features.
- The effect appears before any answer or textual reasoning is generated.
- The improvement extends to targets other than the asked relation and its inverse.

Not established:

- That Qwen had no world-state representation before the question. A failed
  readout cannot establish absence of information.
- That the question creates a complete or stable world state, rather than making
  existing information more accessible, changing its coordinates, or providing
  additional computational positions.
- That Qwen uses these particular features to produce its own answers, or that
  it can answer without accessing the original source trace. The source remains
  available during this experiment's forward pass.
- That these scores match the earlier full-context QA result. That result measured
  Qwen's own generated answers with a different prompt and generation budget.

## Limitations and next checks

1. **Small, previously inspected test set.** There are only four independent
   test layouts; renamings and question variants do not provide new independent
   worlds. Intervals omit selection uncertainty and multiplicity corrections.
   Some bootstrap draws omit a class, and macro then averages present classes.
2. **Nontrivial negative controls.** Question-conditioned shuffled-label macro
   scores reach 58.43%. The real-label banks exceed their matched controls, but
   one train-group permutation is not a null distribution or a significance test.
3. **Question content versus added positions.** There is no neutral continuation
   matched in token length. The question adds both semantic cues and computation
   at additional positions. Mean pooling also changes its token composition.
4. **Different selected readouts.** Each context/target independently selects a
   classifier. The comparison is decodability, not transfer of one fixed decoder
   or a controlled intervention in Qwen's native computation.
5. **Incomplete state schema.** Seven related targets do not reconstruct the
   full world. Probes could learn useful feature combinations without reading
   a pre-existing unified representation.

The next control should compare the spatial question with a neutral continuation
of comparable token length under the same template and selection procedure.
A larger fresh set of independent worlds is needed for stronger generalization
claims. To test causal use, subsequently intervene in Qwen's own activations or
KV-cache and look for directed, property-specific answer changes. Those controls
have not been run here.

## Artifact verification and provenance

- Source report: `artifacts/probes-qwen3-4b/report.json`.
- Predictions: `artifacts/probes-qwen3-4b/test_predictions.jsonl`.
- Feature manifest: `artifacts/features-qwen3-4b/manifest.json`.
- Run commit: `1d8abb581c21f027fecc844b16491af334cc5ca3`.
- Extraction environment: PyTorch 2.15.0+cu132, Transformers 5.22.2.
- All 768 unique context/situation prediction records were checked against the
  dataset targets and groups. Per-target balanced accuracy, mean macro and joint
  accuracy were recomputed for real-label and shuffled-label probes and matched
  the report. Displayed intervals are taken from the report, not independently
  recomputed during this audit.
- All 1,152 feature-shard hashes and all 24 saved prompt-file hashes matched
  their manifest. Dataset and feature manifest hashes matched the report.
- All 112 classifier checkpoints are present (56 real-label, 56 controls).
  Checkpoint forward inference and frozen-Qwen feature extraction were not rerun
  during the results audit.
- Local `extract.py`, `probe.py` and `common.py` hashes match their recorded run
  hashes. Saved prompts have the same assistant suffix, including the empty
  `<think> ... </think>` block inserted by the template; this is not generated
  reasoning.

SHA256 values:

```text
report.json
  dba2a89d7126a433ef5e8433a8ee7a40f2cb1dad82d0cca1d209b7433ad1d010
test_predictions.jsonl
  f936a951a3e352818c3a1c9524d488f0db190e5af1d5df934c227e631ace5230
feature manifest
  dab0f8ac60b188ecd3f6fd0fbc3cb94fa2c70ce056b6e58bcd0bb004d99334f8
dataset manifest
  5e0c428da100d628a92c3414f055ff2ca2c8d51018f2c0b546b8a5843f25693c
```
