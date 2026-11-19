# Results: shallow MLP probes on Qwen3-1.7B hidden features

This document reports the nonlinear follow-up to the issue #12 linear-probe
experiment. The underlying machine-readable results are:

- `artifacts/mlp-probes-qwen3-1.7b/report.json`;
- `artifacts/mlp-fact-order-transfer-qwen3-1.7b/report.json`.

The MLP runs reuse the same frozen Qwen3-1.7B features and leakage-safe data
split as the linear runs. No Qwen parameter is trained. Only the small probe
heads are trained.

## Plain-language summary

Each example describes one world. Seven questions together describe its
derived state: six ask whether one person is left of another, and one asks who
is leftmost. The probe receives a frozen vector produced by Qwen, not the
textual answer. Its job is to recover all seven answers from that vector.

A linear probe can only separate answers with straight decision boundaries.
The MLP used here gets one additional nonlinear transformation and may recover
more complicated patterns. A better selected MLP does not by itself prove
that nonlinearity was necessary, however, because model selection may also
choose a different layer or pooling rule.

The main result is:

> On shuffled inputs, the MLP recovers individual pairwise relations better
> than the linear probe and the shuffled-label control. It still rarely
> recovers the complete seven-answer state, and it fails to recover the
> leftmost person reliably. This is evidence for partial nonlinear
> decodability, not evidence for a complete, stable world-state vector.

In particular, when trained and tested only on shuffled examples, the MLP
reaches 58.5% mean macro accuracy, compared with 48.4% for the linear probe
and 48.4% for the MLP shuffled-label control. Its exact-state accuracy is only
12.2%, however, compared with 10.4% for the control.

## Experimental setup

- Frozen model and features: the same Qwen3-1.7B feature dataset used by the
  linear probes.
- Inputs to Qwen: situation only, without a probe question or expected answer.
- Candidate representations: last situation token and mean-pooled situation
  tokens at blocks 4, 8, 12, 16, 20, 24, and 28.
- Probe: a separate `2048 -> 128 -> classes` MLP for each of the seven
  questions, with GELU and dropout 0.1.
- Optimizer settings: learning rate `1e-3`, at most 200 epochs, and early
  stopping with patience 20.
- Ensemble: five fixed training seeds. Reported predictions average their
  logits.
- Selection: pooling, layer, and weight decay are selected using validation
  data only. The locked configuration is then evaluated on test data.
- Independent split unit: abstract person-rank layout. The test split has only
  four independent layouts, rendered into 768 matched examples.
- Confidence intervals: 95% percentile bootstrap over abstract-layout groups.

As in the linear report, mean macro accuracy first weights every class equally
inside each question and then weights all seven questions equally. Joint exact
state counts an example as correct only if all seven answers are correct at
the same time. A useful random-prediction reference for mean macro accuracy is
about 47.6%.

## Shared MLP across canonical and shuffled inputs

The shared MLP was trained on both fact-order conditions. Validation selection
chose last-token pooling at block 20, hidden size 128, and weight decay
`0.001`.

| Test condition | Mean macro accuracy | Joint exact state |
|---|---:|---:|
| All examples | 75.85% | 45.18% |
| Canonical fact order | 94.92% | 79.95% |
| Shuffled fact order | 56.78% | 10.42% |
| Shuffled-label control, all | 56.92% | 13.67% |
| Shuffled-label control, canonical | 63.08% | 19.53% |
| Shuffled-label control, shuffled | 50.76% | 7.81% |

For all test examples, the 95% group-bootstrap intervals are:

- mean macro accuracy: 73.40% to 78.39%;
- joint exact-state accuracy: 35.03% to 55.21%.

Validation performance was 79.57% mean macro accuracy and 54.69% exact state.
The decrease to 75.85% and 45.18% on test is a reminder that the number of
independent worlds is small despite the larger number of rendered rows.

The shared MLP is clearly successful on canonical inputs. On shuffled inputs,
its macro score is 6.03 percentage points above its matched shuffled-label
control, but exact-state accuracy is only 2.60 points above that control.
Thus it recovers some individual answers without reliably assembling all
seven answers into one correct state.

## Separating nonlinearity from representation selection

The independently selected MLP uses a different representation from the
selected linear probe: last-token block 20 instead of mean-pooled block 16.
The report therefore also trains an MLP on the representation selected by the
shared linear experiment.

| Probe on mean-pooled block 16 | All macro / exact | Canonical macro / exact | Shuffled macro / exact |
|---|---:|---:|---:|
| Linear | 69.11% / 36.20% | 86.86% / 58.07% | 51.36% / 14.32% |
| MLP | 70.42% / 39.97% | 92.05% / 67.97% | 48.78% / 11.98% |

On exactly the same representation, the MLP improves canonical performance
but does not improve shuffled performance. Therefore the strong shared MLP
headline is not caused by nonlinearity alone: selecting the last-token block-20
representation is also important.

## Fact-order-specific training and transfer

Separate MLP banks were selected and trained using only canonical rows or only
shuffled rows. Each locked bank was then evaluated on both test orders.

The table reports mean macro accuracy and joint exact-state accuracy. Values
in parentheses are the corresponding source-condition shuffled-label
controls.

| Train order | Canonical test | Shuffled test |
|---|---:|---:|
| Canonical | **93.18% / 73.44%** (68.46% / 11.72%) | 46.86% / 4.95% (47.59% / 17.45%) |
| Shuffled | 58.49% / 21.35% (54.76% / 14.58%) | **58.49% / 12.24%** (48.44% / 10.42%) |

### Canonical-trained MLP

Validation selected mean pooling at block 16 with weight decay `0.0001`.
Canonical test performance is high and close to the canonical-specific linear
probe. The MLP does not transfer to shuffled inputs: 46.86% macro accuracy is
slightly below its 47.59% control. This is evidence against one readout working
unchanged across the two fact orders.

### Shuffled-trained MLP

Validation selected the last token at block 20 with weight decay `0.0001`.
On shuffled test inputs it reaches:

- mean macro accuracy: 58.49%, 95% CI 53.60% to 61.90%;
- joint exact-state accuracy: 12.24%, 95% CI 5.73% to 18.75%.

The shuffled-label control reaches 48.44% macro and 10.42% exact state. Unlike
the shuffled-specific linear probe, which was at its control, the MLP has a
clear macro-accuracy advantage. This closes one ambiguity from the linear
experiment: shuffled hidden features do contain information that a small
nonlinear readout can recover.

The gain is not uniform across state fields:

| Probe | Shuffled-trained MLP | Shuffled-label control |
|---|---:|---:|
| Bob left of John | 55.2% | 45.6% |
| Bob left of Mary | 65.1% | 47.7% |
| John left of Bob | 55.5% | 48.2% |
| John left of Mary | 65.4% | 53.1% |
| Leftmost person | 37.3% | 40.6% |
| Mary left of Bob | 65.4% | 50.3% |
| Mary left of John | 65.6% | 53.6% |

The MLP mainly recovers pairwise relations, especially relations involving
Mary. It does not beat the control for the three-class leftmost-person field.
This explains how mean macro accuracy can improve while exact recovery of the
whole state remains weak.

The shuffled-trained MLP obtains the same rounded 58.49% macro accuracy on
canonical test inputs, but this is only 3.73 points above the corresponding
54.76% control and has a very wide 33.48% to 80.28% bootstrap interval. It is
not strong evidence of an order-invariant representation.

## What the experiment establishes

The linear result left two possibilities for shuffled inputs:

1. the relevant state information is absent from the tested hidden features;
2. some information is present, but it is not organized so that a linear
   classifier can read it.

The shuffled-specific MLP shows that the first explanation is too strong for
several pairwise relations: the tested hidden features preserve some usable
information. It is consistent with the second explanation, but does not prove
that nonlinearity itself is necessary. The selected MLP used last-token block
20, whereas the selected shuffled-specific linear probe used mean-pooled block
28. A locked comparison of linear and MLP heads on last-token block 20 is
still needed to isolate the effect of the nonlinear head.

It does **not** establish that Qwen has one complete world-state vector:

- only 12.24% of shuffled examples have all seven answers correct;
- the leftmost-person field is not recovered above control;
- a canonical-trained MLP does not transfer to shuffled inputs;
- an MLP can learn task-specific nonlinear computations rather than merely
  read an already prepared state;
- the frozen model's own full-context answers were weak in the earlier
  control, so decodability does not imply causal use.

The appropriate conclusion is therefore:

> Qwen3-1.7B hidden features preserve partial information about shuffled
> spatial facts that is available to a shallow nonlinear readout. The result
> does not demonstrate a complete, stable, order-invariant, or causally used
> world-state representation.

## Limitations

1. There are only four independent test layouts. Rendered variants do not
   create four hundred independent worlds, and several cross-order confidence
   intervals are consequently very wide.
2. The MLP follow-up was designed after the existing linear test results were
   inspected. Its positive finding should be confirmed on newly generated,
   untouched worlds.
3. The MLP has enough flexibility to compute task-specific transformations.
   Better probe accuracy is nonlinear decodability, not proof that Qwen itself
   uses the same computation.
4. The selected and matched-representation results answer different
   questions. The former estimates the best tested MLP configuration; the
   latter isolates the effect of replacing a linear head with an MLP on a
   fixed representation.
5. The current task has redundant reverse pairwise questions and a small set
   of names, rooms, and layouts. A larger synthetic state space is needed for
   a stronger generalization test.

## Recommended next step

Confirm the shuffled-specific result on a newly generated dataset with many
more independent world states and an untouched test split. Pre-register the
last-token block-20 MLP and compare it directly with a linear probe and a
shuffled-label control on the same representation. This avoids another search
over layers and makes the nonlinear-versus-linear claim easier to interpret.
