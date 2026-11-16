# Results: linear probes on Qwen3-1.7B hidden features

This document summarizes the first issue #12 linear-probe run. The underlying
machine-readable result is `artifacts/probes-qwen3-1.7b/report.json`.

## Experimental setup

- Frozen model: Qwen3-1.7B.
- Input: situation only, without a question or chat template.
- Candidate transformer blocks: 4, 8, 12, 16, 20, 24, and 28.
- Representations: last situation token and mean-pooled situation tokens.
- Readout: one shared linear classifier per fixed question across all
  instruction and fact-order conditions.
- Selection: pooling, layer, and L2 were selected using validation mean
  per-probe macro accuracy. Test was evaluated only after selection.
- Selected configuration: mean pooling, block 16, L2 = `1e-5`.
- Independent split unit: abstract person-rank layout. The test split contains
  4 independent layouts and 768 rendered examples.
- Confidence intervals: 95% percentile bootstrap over abstract-layout groups.

The dataset manifest SHA-256 recorded by the result matches the local dataset:
`c308a3b59c0ebe9c714cfe756bba9f5cbf1c9269a329f6d6f500c0dca569e39f`.
The recorded feature-manifest SHA-256 is
`bdac9fca29b594a57cccb7f48df66762e7c52b26cb9e44ae2ca2918c53ed3198`.

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

## Main results

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

## Interpretation

The run provides evidence that Qwen3-1.7B hidden features contain linearly
decodable spatial relations under a regular canonical presentation. It does
not provide evidence for a stable, order-invariant world-state representation.

The strongest observation is the fact-order gap:

- canonical mean macro accuracy: 86.9%;
- shuffled mean macro accuracy: 51.4%;
- canonical exact state: 58.1%;
- shuffled exact state: 14.3%.

Because the canonical and shuffled inputs contain identical facts, this gap
shows that the shared linear readout is strongly dependent on how those facts
are arranged. In the canonical format, room relations form a left-to-right
chain and person facts use a fixed name order. The probe can exploit hidden
features aligned with that regular sequence. After permutation, performance is
near the random reference and only slightly above the matched shuffled-label
control.

The appropriate conclusion is therefore:

> Frozen Qwen3-1.7B hidden features contain format-dependent, linearly
> decodable spatial information, but this experiment does not establish a
> stable or causally usable world-state representation.

## Limitations

1. Only 24 abstract layouts exist, with 3 validation and 4 test layouts. The
   independent sample size is much smaller than the rendered row count, and
   the confidence intervals are correspondingly wide.
2. A shared probe across fact-order conditions tests whether one aligned
   linear readout transfers between conditions. Failure on shuffled text does
   not distinguish information loss from a condition-specific rotation or
   reorganization of the representation.
3. The shuffled-label control is above the theoretical random reference on
   some probes, consistent with high variance from the small structured split.
   Direct differences from the matched control are more informative than raw
   accuracy alone.
4. The run tests decodability, not causal use. It does not remove source tokens
   and feed a hidden-state bottleneck to a frozen reader.
5. Full-context Qwen QA and lexical/surface baselines have not yet been
   reported, so task solvability and surface-feature contributions remain to
   be quantified.
6. The result directory currently preserves the report and selected probe, but
   not the feature manifest or feature shards needed to audit extraction
   provenance and reproduce fitting from the saved activations.

## Recommended next steps

1. Train and evaluate separately labelled canonical-only and shuffled-only
   probes. A strong shuffled-only probe would show that the state remains
   decodable but is not aligned with the canonical representation; a weak one
   would support genuine loss of linear decodability.
2. Run the frozen full-context Qwen QA control and a lexical/surface baseline
   on the same splits.
3. Expand the synthetic task so validation and test contain substantially more
   independent world states, names, locations, and event compositions.
4. Add paired group-level intervals for fact-order and instruction contrasts.
5. After the decodability controls are complete, test causal use through a
   matched source-free interface with real, shuffled, counterfactual-swapped,
   null, and oracle state inputs.
6. Preserve the feature manifest and immutable feature artifacts together with
   the report, selected probe, repository commit, and run log.
