# Results: hidden-state probes and full-context Qwen QA

Updated 2026-11-25. The probe and fact-order transfer results below are for
**Qwen3-1.7B only**. Full-context QA now includes Qwen3-1.7B and Qwen3-4B,
with separate canonical and shuffled runs.

Archived probe artifacts:

- `artifacts/prev/probes-qwen3-1.7b/report.json`;
- `artifacts/prev/fact-order-transfer-qwen3-1.7b/report.json`;
- the original combined QA run at
  `artifacts/prev/full-context-final-qwen3-1.7b/report.json`.

The current full-context results come from `report.json` and `responses.jsonl`
in each of these directories:

- `artifacts/full-context-canonical-qwen3-1.7b/`;
- `artifacts/full-context-shuffled-qwen3-1.7b/`;
- `artifacts/full-context-canonical-qwen3-4b/`;
- `artifacts/full-context-shuffled-qwen3-4b/`.

## Experimental setup

- Frozen model: Qwen3-1.7B.
- Input: situation only, without a question or chat template.
- Candidate transformer blocks: 4, 8, 12, 16, 20, 24, and 28.
- Representations: last situation token and mean-pooled situation tokens.
- Shared readout: one linear classifier per fixed question across all
  instruction and fact-order conditions. Its selected configuration was mean
  pooling, block 16, L2 = `1e-5`.
- Fact-order transfer readouts: one probe bank trained only on canonical rows
  and one trained only on shuffled rows. Pooling, layer, and L2 were selected
  independently using validation rows of the matching source condition only.
  Both test conditions were evaluated after each source probe was locked.
- Independent split unit: abstract person-rank layout. The test split contains
  4 independent layouts and 768 rendered examples.
- Confidence intervals: 95% percentile bootstrap over abstract-layout groups.

The dataset manifest SHA-256 recorded by the probe result matches the archived
`artifacts/prev/db_prev/manifest.json`:
`c308a3b59c0ebe9c714cfe756bba9f5cbf1c9269a329f6d6f500c0dca569e39f`.
The recorded feature-manifest SHA-256 is
`bdac9fca29b594a57cccb7f48df66762e7c52b26cb9e44ae2ca2918c53ed3198`.

Current QA uses the v2 export in `artifacts/db`. Its JSONL hashes match all four
QA reports. The `id`, `text`, `targets`, and `group_id` fields are identical
between the archived and current database across all three splits; v2 adds
explicit fact lists and QA metadata. The different export hashes do not imply
a change to the situations or labels evaluated here.

## Conditions

The canonical and shuffled examples contain exactly the same six facts. Only
their order differs.

In the canonical condition, the three adjacent room relations are presented
in left-to-right order, followed by the locations of Mary, John, and Bob in
that fixed name order. For example:

```text
Room D is left of room B.
Room B is left of room A.
Room A is left of room C.
Mary is in room D.
John is in room B.
Bob is in room A.
```

In the shuffled condition, those same sentences are deterministically
permuted for each episode. The final world state is unchanged, but recovering
it requires integrating facts that no longer form a regular textual sequence.

## Metrics

For each probe, macro accuracy is the unweighted mean of its per-class recall.
The reported mean macro accuracy is then the unweighted mean across all seven
probes. It measures how accurately individual fields and relations are read
without allowing frequent classes or questions to dominate the score.

Joint exact-state accuracy counts an example as correct only when all seven
probe answers are correct simultaneously. The reverse pairwise questions are
logically redundant, so this metric represents exact recovery of the induced
ordering rather than seven independent state fields.

The reference mean macro accuracy for uniformly random predictions is about
47.6%: six binary probes at 50% and one three-class probe at 33.3%.

## Shared-probe results

| Test condition | Examples | Mean macro accuracy | Joint exact state |
|---|---:|---:|---:|
| All examples | 768 | 69.1% | 36.2% |
| Canonical fact order | 384 | 86.9% | 58.1% |
| Shuffled fact order | 384 | 51.4% | 14.3% |
| Shuffled-label control, all | 768 | 54.5% | 9.1% |
| Shuffled-label control, canonical | 384 | 62.7% | 7.6% |
| Shuffled-label control, shuffled | 384 | 46.3% | 10.7% |

For all test examples, the 95% group-bootstrap intervals are:

- mean macro accuracy: 63.9% to 74.5%;
- joint exact-state accuracy: 24.2% to 48.2%.

Validation mean macro accuracy was 73.3%, compared with 69.1% on test.
Validation exact-state accuracy was 48.8%, compared with 36.2% on test.

### Per-probe test results

| Probe | Test macro | Shuffled-label control | Difference |
|---|---:|---:|---:|
| Bob left of John | 65.8% | 67.4% | -1.7 pp |
| Bob left of Mary | 74.5% | 38.5% | +35.9 pp |
| John left of Bob | 65.8% | 67.4% | -1.7 pp |
| John left of Mary | 74.3% | 55.9% | +18.5 pp |
| Leftmost person | 54.6% | 57.8% | -3.2 pp |
| Mary left of Bob | 74.5% | 38.5% | +35.9 pp |
| Mary left of John | 74.3% | 55.9% | +18.5 pp |

The overall improvement is therefore not uniform across state fields. Much of
the reliable gain over the shuffled-label control comes from relations that
involve Mary. The Bob-John relation and leftmost-person probe do not beat this
particular control on the held-out layouts.

## Instruction ablation

| Instruction | Mean macro accuracy | Joint exact state |
|---|---:|---:|
| No instruction | 68.8% | 34.4% |
| `Imagine the situation.` | 69.2% | 36.5% |
| `Think about who is where.` | 68.0% | 36.5% |
| `Think about where Mary is.` | 70.5% | 37.5% |

Instruction differences are small compared with the fact-order effect. On
shuffled examples, the Mary-specific instruction gives modestly higher scores
on Mary-related probes, but the test split has only four independent layouts
and no paired confidence interval was computed for this contrast. It should be
treated as suggestive rather than conclusive evidence of selective state
preparation.

## Layer and pooling selection

The selected block-16 mean-pooled representation achieved 73.3% validation
mean macro accuracy. The best block-8 last-token candidate achieved 72.8%, and
the best block-8 mean-pooled candidate achieved 72.7%. These differences are
too small to support a strong claim that block 16 uniquely contains the state
or that mean pooling is decisively superior.

Mean pooling also retains information distributed over the contextualized
token trace. Success with this representation is weaker evidence for a compact
prepared state than success at one final token or through a source-free fixed
bottleneck.

## Fact-order transfer diagnostic

The shared run could not distinguish two explanations for weak shuffled
performance: the state might be absent from the tested representation, or it
might remain linearly decodable in a different condition-specific coordinate
system. The transfer diagnostic therefore fitted and selected probes
separately for canonical and shuffled train rows, then evaluated each locked
probe on both test orders.

The table reports mean macro accuracy and joint exact-state accuracy. The
parenthesized values are the corresponding source-condition shuffled-label
controls.

| Train order | Canonical test | Shuffled test |
|---|---:|---:|
| Canonical | **93.0% / 72.1%** (65.6% / 16.7%) | 45.1% / 11.5% (44.1% / 12.8%) |
| Shuffled | 48.5% / 13.0% (50.8% / 12.5%) | **48.4% / 9.6%** (47.5% / 12.2%) |

### Canonical-specific probe

The canonical-only selection chose mean pooling at block 8 with L2 = `0.001`.
It achieved 100% validation macro and exact-state accuracy. On canonical test
rows it achieved:

- mean macro accuracy: 93.0%, 95% CI 87.4% to 98.4%;
- joint exact-state accuracy: 72.1%, 95% CI 53.9% to 89.6%.

This is stronger than the shared probe on canonical rows (86.9% macro and
58.1% exact state), showing that mixing shuffled rows into a single shared
readout reduced the best canonical decodability estimate.

The canonical-only probe did not transfer to shuffled rows: 45.1% macro and
11.5% exact state, essentially matching its shuffled-label control at 44.1%
and 12.8%.

### Shuffled-specific probe

The shuffled-only selection chose mean pooling at block 28 with L2 = `0.001`.
Even its selected validation result was weak: 52.3% macro and 16.7% exact
state. On shuffled test rows it achieved:

- mean macro accuracy: 48.4%, 95% CI 45.6% to 50.9%;
- joint exact-state accuracy: 9.6%, 95% CI 6.3% to 15.6%.

The matched shuffled-label control achieved 47.5% macro and 12.2% exact state.
Thus the shuffled-specific probe provided no meaningful improvement over the
negative control. The best validation candidates across other layers and both
pooling rules were also clustered near 52% macro, so the failure is not
explained by selecting one unusually bad layer.

The shuffled-only probe also did not transfer to canonical rows: 48.5% macro
and 13.0% exact state, compared with its control at 50.8% and 12.5%.

### Instructions within the condition-specific runs

Instructions did not rescue shuffled-specific decodability:

| Shuffled train and test instruction | Mean macro accuracy | Joint exact state |
|---|---:|---:|
| No instruction | 53.0% | 11.5% |
| `Imagine the situation.` | 47.0% | 8.3% |
| `Think about where Mary is.` | 50.0% | 12.5% |
| `Think about who is where.` | 43.5% | 6.3% |

With only four independent test layouts, these differences are consistent
with noise rather than a reliable instruction effect.

## Full-context frozen-model QA control

Each prompt contains the full situation and one fixed question, rendered with
Qwen's chat template. Both models are frozen, `enable_thinking=false`, and
generation is greedy. The requested output is `FINAL: <answer>`. Ground truth
is used only for evaluation. The scorer accepts the last recognized FINAL
marker, with a leading short-answer fallback; invalid responses count as errors.

All four current reports record schema `spatial-full-context-final-answer-v3`
and repository commit `85efb9a2a036a3652edb0d6c56bedc684029744d`.
Model revisions are:

- Qwen3-1.7B: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
- Qwen3-4B: `1cfa9a7208912126459214e8b04321603b3df60c`.

Each run has 288 validation and 384 test situations, covering all four
instruction variants: 2,016 validation and 2,688 test responses. Situation,
question, expected answer, and dataset hashes were checked against the database.

### Recorded results

| Model | Fact order | Max new tokens | Validation macro | Validation joint exact | Test macro | Test joint exact |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Canonical | 64 | 51.03% | 0.00% (0/288) | 51.22% | 0.00% (0/384) |
| Qwen3-1.7B | Shuffled | 64 | 53.39% | 0.69% (2/288) | 54.10% | 1.82% (7/384) |
| Qwen3-4B | Canonical | 2048 | 94.01% | 70.49% (203/288) | 93.07% | 64.58% (248/384) |
| Qwen3-4B | Shuffled | 2048 | 96.99% | 79.51% (229/288) | 95.00% | 73.96% (284/384) |

These are the original report metrics, without post-hoc parser changes.
The 4B test group-bootstrap 95% intervals are:

| Fact order | Mean macro accuracy | Joint exact state |
|---|---:|---:|
| Canonical | 89.36–96.80% | 44.79–86.98% |
| Shuffled | 92.11–97.84% | 61.46–88.54% |

Only four independent test layouts underlie these scores. The observed
shuffled advantage is descriptive; no paired group-level interval was computed
for the order contrast.

### Generation budget and formatting

The prompt and scoring rules are shared, but the current cross-model comparison
is **not a fully matched protocol**: 1.7B uses a 64-token generation budget and
batch size 16, while 4B uses 2048 tokens and batch size 8. A 1.7B run with the
same larger generation budget is still needed to isolate model-size effects.

The earlier 4B shuffled run with 64 tokens had 2,655/2,688 invalid test
responses and 1.04% test macro accuracy. Its raw outputs typically stopped
mid-explanation without a FINAL answer. That run was inspected before the
current 2048-token artifacts replaced it at the same directory; these older
numbers are historical observations, not metrics recoverable from the current
report. The 2048-token run shows that the earlier score was dominated by the
answer-generation budget, not evidence that 4B could not solve the task.
Future reruns should use distinct output directories for each budget.

Thinking is disabled in all reports above, but 4B still writes explanations
in ordinary output text. Disabling the template's thinking mode does not mean
that every response is a single short answer.

| Model / order | Invalid validation responses | Invalid test responses |
|---|---:|---:|
| 1.7B / canonical | 0/2,016 | 0/2,688 |
| 1.7B / shuffled | 0/2,016 | 0/2,688 |
| 4B / canonical | 16/2,016 | 26/2,688 |
| 4B / shuffled | 30/2,016 | 51/2,688 |

Some remaining 4B invalids are parser limitations: formats such as
`### FINAL: Yes` and `**Answer: FINAL: No**` are rejected. A diagnostic scan
found recoverable allowed FINAL answers in all 16 canonical validation
invalids and 20/26 canonical test invalids, and in all 30 shuffled validation
invalids and 42/51 shuffled test invalids. This does not mean those recovered
answers are all correct. The official scores above retain the original parser;
any revised scoring should be versioned and applied uniformly to every run.

### Answer behavior

For 1.7B, canonical test answers show a strong Yes bias: 2,280/2,304 binary
answers are Yes (98.96%). All 384 canonical situations and 370/384 shuffled
situations have at least one inconsistent reverse pair. The lack of rejected
formats does not explain this model's weak scores under the tested protocol.

For 4B, among situations where all seven answers were parsed, 95/360 canonical
and 32/338 shuffled test situations contain an inconsistent reverse pair.
Some explanations correctly list the room and person order, then reverse a
relation or answer the inverse question. These are examples of failure to use
an explicitly generated intermediate arrangement correctly, not direct evidence
about a pre-existing situation-only hidden representation.

The leftmost-person question remains difficult in some cases: test accuracy
is 89.84% canonical and 87.50% shuffled. In several shuffled responses the model
confuses the leftmost person with the occupant of the absolute leftmost room,
even when that room is empty, and continues without a final answer.

4B provides a strong full-context solvability control for both fact orders
with sufficient output budget. It does not establish that 1.7B can solve the
task, nor does it supply the missing 4B situation-only probe or causal-use runs.

## Interpretation

The probe runs provide strong evidence that Qwen3-1.7B hidden features contain
linearly decodable spatial relations under a regular canonical presentation.
They do not provide evidence for a stable, order-invariant world-state
representation. The full-context control additionally shows that high
canonical linear decodability does not imply that the frozen model will use
that information to answer the fixed questions correctly under this protocol.

Within the 1.7B probe results, the main observation is the condition-specific
fact-order gap:

- canonical-to-canonical mean macro accuracy: 93.0%;
- shuffled-to-shuffled mean macro accuracy: 48.4%;
- canonical-to-canonical exact state: 72.1%;
- shuffled-to-shuffled exact state: 9.6%.

The transfer diagnostic tests whether training on shuffled inputs rescues
linear decoding within the selected representation and probe family. It does
not exclude every possible linear readout or a nonlinear representation. Here, shuffled-to-shuffled performance remained at the
random reference and matched its shuffled-label control. Both cross-order
directions also failed.

Because the canonical and shuffled inputs contain identical facts, the result
shows that linear state decodability is strongly dependent on how those facts
are arranged. In the canonical format, room relations form a left-to-right
chain and person facts use a fixed name order. The probe can exploit hidden
features aligned with that regular sequence. After permutation, none of the
tested layers or pooling rules exposes a comparably integrated linear state.

The appropriate conclusion is therefore:

> Frozen Qwen3-1.7B hidden features contain a strongly linearly decodable
> spatial state under canonical presentation. The same state is not reliably
> linearly decodable after fact-order permutation, even with a probe trained
> specifically for shuffled inputs. The experiment therefore does not
> establish a stable, order-invariant, or causally usable world-state
> representation.

The new 4B control narrows the interpretation: the shuffled task can be solved
from full context by this larger model with a longer output budget. It does not
turn the 1.7B shuffled probe failure into evidence of absent state. Probes must
be measured on 4B itself before comparing its native QA with its decodability.

The hypothesis that a model encodes useful information but cannot reliably
apply it remains consistent with issue #12. Strong canonical 1.7B probes and
weak native QA distinguish external linear readability from native answer
behavior. They do not, without surface baselines and causal interventions,
prove a coherent world model that the model itself can use. Full-context 4B
reasoning may recompute the answer from the source trace.

## Limitations

1. Only 24 abstract layouts exist, with 3 validation and 4 test layouts. The
   independent sample size is much smaller than the rendered row count, and
   the confidence intervals are correspondingly wide.
2. The condition-specific diagnostic found no successful shuffled readout
   within the tested probe family, but it does not show that all state
   information is absent. The information could be nonlinearly decodable or
   distributed in a representation not tested here.
3. The shuffled-label control is above the theoretical random reference on
   some probes, consistent with high variance from the small structured split.
   Direct differences from the matched control are more informative than raw
   accuracy alone.
4. The run tests decodability, not causal use. It does not remove source tokens
   and feed a hidden-state bottleneck to a frozen reader.
5. Native QA is weak for 1.7B at 64 tokens but strong for 4B at 2048 tokens.
   Generation budgets are not matched across models, and no 4B probes have
   been reported here. Lexical/surface baselines are still missing.
6. The result directories currently preserve the reports and selected probes,
   but not the feature manifest or feature shards needed to audit extraction
   provenance and reproduce fitting from the saved activations.

## Recommended next steps

1. Run a lexical/surface baseline and, as a secondary diagnostic, a small MLP
   probe on the same frozen features. The MLP must remain clearly labelled as
   nonlinear decodability rather than evidence for a compact linear state.
2. Match the 1.7B/4B generation budgets and apply any parser correction
   uniformly, retaining original reports. Run situation-only feature extraction
   and probes on 4B, then compare decoding and full-context QA within that
   same model. Keep thinking-enabled runs as a separate protocol.
3. Expand the synthetic task so validation and test contain substantially more
   independent world states, names, locations, and event compositions.
4. Add paired group-level intervals for fact-order and instruction contrasts.
5. After the decodability controls are complete, test causal use through a
   matched source-free interface with real, shuffled, counterfactual-swapped,
   null, and oracle state inputs.
6. Preserve the feature manifest and immutable feature artifacts together with
   the report, selected probe, repository commit, and run log.
