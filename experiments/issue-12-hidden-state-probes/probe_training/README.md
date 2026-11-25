# Qwen3-4B: instruction-specific linear probes

Пошаговый запуск на существующей базе: [RUNBOOK.md](RUNBOOK.md).

This follow-up trains **56 separate linear classifiers**: seven fixed questions
× four instructions × two fact orders. Every classifier independently selects
its pooling, layer and L2 on validation examples from its own condition. It
reuses the original dataset and feature extraction format; earlier shared-bank
results remain a distinct experiment.

## Dataset audit and suitability

The existing `../artifacts/db` is suitable for this **exploratory relative-order
probe experiment**, without regeneration. The audit performed on 2026-11-25 found:

| Split | Abstract-layout groups | Episodes | Rows per instruction/order cell | All rows |
|---|---:|---:|---:|---:|
| Train | 17 | 408 | 408 | 3,264 |
| Validation | 3 | 72 | 72 | 576 |
| Test | 4 | 96 | 96 | 768 |

All episodes have all eight matched conditions, identical state/targets, and
identical facts across instructions within each order. Groups and episodes do
not cross splits. Every question has all classes in every split/cell. The
script checks these properties and dataset hashes before training; feature
hashes and ID joins are checked too.

The row counts overstate independent semantic diversity: room renamings do
not create new abstract layouts. Three validation groups are a weak basis for
selecting many hyperparameters; four test groups give unstable, exploratory
intervals. Adding more room-name permutations cannot fix this. Keep the grid
small, publish every candidate score, and do not tune it against these test
results. The existing test has already been inspected in earlier experiments;
this is a follow-up on that benchmark, not a fresh confirmatory holdout.

For stronger generalization claims, design a larger world family (more rooms,
people, event compositions) with enough independent held-out states, then lock
new train/validation/test splits. New names require an explicit label mapping
or entity-query design; the current seven fixed-name heads do not test that.

The existing targets encode relative person order, **not full world state**:
room assignment, absolute rank and updates are not fully reconstructed. The
stored `state.person_rank` and `state.person_room` could support additional
person-specific probes later, but four rank classes cannot all occur in only
three validation layouts. Such an extension needs redesigned splits. No
causal-use or source-removal claim follows from this experiment.

## Protocol

- Extract frozen **Qwen/Qwen3-4B** situation-only features with the pinned
  revision used by the full-context QA control. No question or chat template.
- For each instruction/order cell and question, fit multinomial linear heads
  on train only, with train-only feature standardization.
- Select `(pooling, layer, L2)` using that question's validation balanced
  accuracy (mean class recall). Ties: ordinary accuracy, stronger L2, then
  first candidate in the specified pooling/layer order. LBFGS iteration budget
  is fixed, not selected. Neither weights nor normalization use validation.
- Evaluate the selected head on test, without refitting on train+validation.
  Assemble the independently selected predictions into a bank for joint
  exact-target accuracy. This is exactness of the seven targets, not the full
  room/person state.
- Fit a negative control at the selected configuration, permuting complete
  target vectors among train groups. This is one fixed-seed sanity check,
  not a permutation significance test or separately optimized null baseline.
- Compare instruction-specific banks against `none`, separately for each fact
  order. Compare independently trained shuffled/canonical banks as well.
  These comparisons measure best selected decodability; they do not measure
  transfer of a common readout or isolate the coordinate system of features.

## Selective Mary effect

Predefined families:

- Mary relations: Mary–Bob and Mary–John, both directed questions.
- Non-Mary relations: Bob–John, both directed questions.
- `leftmost_person`: reported separately; it involves all three people.

For `track_mary` versus `none`, report `ΔMary`, `ΔBobJohn`, and
`ΔMary − ΔBobJohn`. All deltas use mean per-question balanced accuracy. The
last quantity is the selective-effect interaction. A positive interaction
alone is insufficient: inspect whether Mary improves and whether Bob/John
improve, stay similar, or degrade. An interval containing zero does not prove
that an effect is absent. Reverse-direction questions are logically redundant
and are not independent observations. Mary-family relations also involve the
other person; these are relational selectivity measures, not isolated Mary
position decoding.

Intervals use paired percentile bootstrap over test `group_id`: align episodes,
resample identical groups for both conditions and all questions, and compute
*differences inside each bootstrap replicate*. The same method reports
per-question, aggregate and joint-target differences. Class recall is averaged
over classes present in each draw, matching the existing metrics implementation;
with four groups a draw can omit a class. Intervals are conditional on the
trained/selected probes, omit training/selection uncertainty, and are not
multiplicity-adjusted. Treat the predeclared `track_mary` interaction as primary
and the remaining comparisons as exploratory.

## Commands (from repository root)

Audit only (does not load Qwen or train):

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/probe_training/train.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --audit-only
```

Extract Qwen3-4B features into this follow-up's own artifact folder. The existing
extractor accepts the model/revision explicitly; do not reuse 1.7B features.
The pinned weights must already be in the local Hugging Face cache (see the
parent [runbook](../RUNBOOK.md)).

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 4,8,12,16,20,24,28,last \
  --device cuda --model-dtype bfloat16 --output-dtype float16 --batch-size 4
```

Train and evaluate:

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/probe_training/train.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/features-qwen3-4b \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/probes-qwen3-4b \
  --poolings last_token,mean_pool,sentence_end_mean \
  --l2-values 0,1e-5,1e-4,1e-3 \
  --max-iter 100 --device cpu --threads 1 --bootstrap-samples 2000
```

The output directory must not exist. It contains 56 checkpoints (weights,
standardization, selected configuration and class mapping), `report.json`
(candidate scores, cell metrics, negative control and paired effects), and
`test_predictions.jsonl` for auditing. Failed runs can leave partial checkpoints;
use a new output directory for a retry. No report is written until completion.

The QA reference in [RESULTS.md](../RESULTS.md) is strong but imperfect for 4B:
93.07%/95.00% test mean macro accuracy for canonical/shuffled facts, versus
64.58%/73.96% joint-target accuracy. Compare probes with the **same model and
conditions**, and do not restrict probe evaluation to QA-successful examples.

Implementation validation uses synthetic tensors only. Full Qwen3-4B feature
extraction and probe training are separate runs; no measured probe result is
claimed by adding this folder.
