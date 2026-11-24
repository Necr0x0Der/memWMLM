# Issue #12: spatial hidden-state probe dataset

This folder implements the data-generation part of the proposed first
experiment in [issue #12](https://github.com/Necr0x0Der/memWMLM/issues/12):
linear probes over frozen Qwen3-1.7B hidden features after situation-only
encoding.

For a clean-machine, end-to-end execution guide, see [`RUNBOOK.md`](RUNBOOK.md).

## What the dataset measures

Each world has four linearly ordered rooms and three people.  The text states
the room order and each person's room.  Probe labels describe the derived
world state:

- which person is leftmost;
- every directed pairwise `person_a left of person_b` relation.

These labels require combining the room-order facts with person placements.
They are not included as sentences in the situation.  A linear probe can be
trained for each fixed question at every selected model layer.  This establishes
linear decodability only; it does not establish causal use of the state.

Every semantic episode has matched variants for:

- no instruction;
- `Imagine the situation.`;
- `Think about who is where.`;
- the selective instruction `Think about where Mary is.`;
- canonical and shuffled fact order.

The `text` field contains the complete situation-only encoder input. Each row
also stores `facts` (the order rendered in `text`), plus the matched
`canonical_facts` and `shuffled_facts` lists. The latter is guaranteed to be a
non-identity permutation of the former. `qwen_qa` contains every question,
allowed short answers, and the expected answer, so a shuffled row is directly
usable for full-context Qwen QA by combining its `text` with a selected
`qwen_qa[].question`. Questions must not be appended when extracting the
features used by the primary probes.

For example, a Qwen QA caller can select a row with
`fact_order == "shuffled"`, use `row["text"]` as the situation, and select a
question from `row["qwen_qa"]`. Each question item has this shape:

```json
{
  "id": "mary_left_of_john",
  "question": "Is Mary left of John?",
  "allowed_answers": ["No", "Yes"],
  "answer": "Yes"
}
```

The `answer` is evaluation metadata and must not be included in the Qwen input.

## Leakage-safe split

The split unit is the **abstract rank layout** of Mary, John, and Bob, not an
individual rendered text.  For example, all worlds in which Mary has rank 0,
John rank 3, and Bob rank 2 belong to one group even when room names, fact
order, or instructions differ.  A group is assigned wholly to train,
validation, or test.

This is intentionally stronger than random row splitting: renaming rooms from
`A/B/C/D` to another permutation does not turn the same derived world state
into an independent test example.  The generator also balances all probe
classes in every split.

There are only 24 possible abstract layouts for three people in four rooms, so
70/15/15 cannot be represented exactly.  The deterministic largest-remainder
allocation produces 17/3/4 groups (70.8%/12.5%/16.7%).  The exact counts and
hashes are recorded in the manifest.

## Generate

```bash
python experiments/issue-12-hidden-state-probes/prepare_spatial_probe_dataset.py \
  --output-dir datasets/issue-12-spatial-probes \
  --seed 12 \
  --episodes-per-group 24
```

The output is:

```text
manifest.json
train.jsonl
validation.jsonl
test.jsonl
```

Each of the 24 room-label permutations is used once per abstract layout with
the default `--episodes-per-group 24`.  Each episode expands to eight matched
instruction/order conditions, for 4,608 rows total.

## Extract frozen Qwen features

Qwen3-1.7B has 28 transformer blocks and hidden size 2048 at the pinned model
revision.  The extractor reads only the situation `text`, does not apply a chat
template, does not append a question, and never calls `generate`.  The model is
put in evaluation mode, every parameter has `requires_grad=False`, and the
forward pass runs under `torch.inference_mode()`.
Loading follows the existing Qwen experiment scripts via pinned
`AutoModelForCausalLM.from_pretrained(..., dtype=..., local_files_only=True)`.
The extractor calls only the wrapper's base `.model`, so the language-model
head is not executed and no next-token logits are computed.

```bash
python experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/features-sentence-end-qwen3-1.7b \
  --layers 4,8,12,16,20,24,last \
  --device cuda \
  --model-dtype bfloat16 \
  --output-dtype float16 \
  --batch-size 8
```

Layer numbers are one-based transformer-block numbers: `hidden_states[0]` is
the embedding output, `hidden_states[4]` is the output of block 4, and `last`
resolves to the block-28 output after Qwen's final normalization. Duplicate
selections are removed.

For each shard, the extractor saves:

- `last_token`: hidden state at the last non-padding situation token;
- `mean_pool`: arithmetic mean over all non-padding input tokens;
- `sentence_end_mean`: arithmetic mean of the hidden vectors at the final
  tokenizer token of each of the six fact lines, excluding the instruction;
- `ids`, `token_counts`, `sentence_end_counts`, and `layer_indices` needed to
  join and audit features.

All three feature tensors have shape `[examples, selected_layers, 2048]`. The
extractor refuses silent truncation, verifies dataset hashes against its
manifest, uses the immutable model revision already established by issue #5,
and writes model/tokenizer/pooling provenance to a feature manifest.

The three pooling rules answer slightly different questions. `last_token`
tests whether the causal stream has accumulated state by the end of the
situation. `mean_pool` tests whether state is linearly available across the
complete token trace. `sentence_end_mean` gives each completed fact one causal
summary vector and averages those six vectors, rather than weighting long facts
by their token count. Both mean variants are weaker evidence for a compact
prepared state than success at the last token.

## Intended probe protocol

1. Freeze the tokenizer and model revision before feature extraction.
2. Encode `text` without any question and save the chosen token position or
   pooling rule at every selected layer.
3. Train one linear classifier per manifest probe using only `train.jsonl`.
4. Select regularization, layer, pooling, and stopping choices on
   `validation.jsonl` only.
5. Evaluate the locked choice once on `test.jsonl`.
6. Report per-probe macro accuracy and the joint exact-state accuracy, both by
   instruction and fact-order condition.

For the instruction ablation, pre-register which of two different questions is
being asked.  Separate probes trained and selected within each condition measure
the best linear decodability available under that instruction.  One shared probe
trained across conditions measures whether the same linear readout remains
aligned.  Both are useful, but their scores should not be interpreted as the
same quantity.

The matched variants from one episode must not be treated as independent when
computing confidence intervals; resample at least by `group_id` (or use the
more conservative abstract-layout groups directly).

## Train and evaluate the linear probes

`train_linear_probes.py` implements the shared-readout version of that
protocol.  It trains one multinomial linear classifier per manifest question,
while using one common `(pooling, layer, L2)` choice for the complete probe
bank.  The choice is made by mean per-probe macro accuracy on validation;
joint exact-state accuracy and then stronger L2 break exact ties.  Test is
evaluated only after selection.

```bash
python experiments/issue-12-hidden-state-probes/train_linear_probes.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/artifacts/features-sentence-end-qwen3-1.7b \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/probes-sentence-end-qwen3-1.7b \
  --poolings last_token,mean_pool,sentence_end_mean \
  --l2-values 0,1e-5,1e-4,1e-3 \
  --device cpu
```

The output contains `selected_probe.pt` and `report.json`.  The report includes
validation selection scores, locked validation/test metrics, breakdowns by
instruction and fact order (including their cross-product), and a shuffled
label control.  Headline 95% intervals resample abstract-layout groups rather
than treating matched text variants as independent. Shuffling permutes complete state vectors between train
abstract-layout groups, preserving all matched variants and the marginal label
counts.  Feature shard and dataset hashes are checked before fitting.

This first implementation deliberately does not train separate probes inside
each instruction condition: doing so answers a different, condition-specific
decodability question and adds another model-selection axis.  It can be added
as a separately labelled follow-up after the shared readout is evaluated.

## Fact-order transfer follow-up

`analyze_fact_order_transfer.py` separates condition-specific decodability
from readout alignment. It fits two probe banks: one using only canonical train
rows and one using only shuffled train rows. Each bank selects its own pooling,
layer, and L2 using validation rows of the same source condition only. The
locked bank is then evaluated on both test conditions, producing:

```text
canonical -> canonical    canonical -> shuffled
shuffled  -> canonical    shuffled  -> shuffled
```

This distinction matters because a weak shared probe on shuffled rows could
mean either that state information is not linearly decodable there or that it
is encoded in a condition-specific linear coordinate system.

```bash
python experiments/issue-12-hidden-state-probes/analyze_fact_order_transfer.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/artifacts/features-sentence-end-qwen3-1.7b \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/fact-order-transfer-sentence-end-qwen3-1.7b \
  --poolings last_token,mean_pool,sentence_end_mean \
  --l2-values 0,1e-5,1e-4,1e-3 \
  --device cpu
```

The output contains `report.json`, `probe-trained-canonical.pt`, and
`probe-trained-shuffled.pt`. Each of the four cells includes per-probe and
joint exact-state metrics, instruction breakdowns, group-bootstrap intervals,
and the source-condition shuffled-label control. The original shared-probe
report remains unchanged and should be retained as a separate result.

## Full-context Qwen QA control

`evaluate_full_context_qa.py` asks the frozen Qwen model each manifest question
with the complete situation still in context. This stage does not train a
probe. The prompt asks for exactly one allowed short answer (`Yes`/`No` or one
person name), is rendered through Qwen's chat template with thinking disabled,
and uses deterministic greedy generation. The model is asked to end with
`FINAL: <answer>`; ground truth is used only after generation to compute
accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  experiments/issue-12-hidden-state-probes/evaluate_full_context_qa.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/artifacts/full-context-final-qwen3-1.7b \
  --splits validation,test \
  --device cuda \
  --model-dtype bfloat16 \
  --batch-size 16 \
  --max-length 256 \
  --max-new-tokens 64
```

The output contains `report.json` and `responses.jsonl`. The latter preserves
every raw model response, parsed class, target, and condition for audit. The
parser prefers the last `FINAL:` marker and also accepts an unambiguous leading
short answer such as `Yes` or `Answer: **Yes**`; other format violations count
as incorrect.
Metrics use the same per-question macro accuracy, joint exact-state accuracy,
instruction/fact-order breakdowns, and group-bootstrap intervals as the probe
reports.

## Required interpretation and controls

High linear-probe accuracy means that the tested features make the state
linearly decodable.  It does not show that the frozen LLM uses this state.
The execution protocol should also report shuffled-label probes, a lexical
surface baseline, and full-context Qwen accuracy.  A later causal-use stage
still needs source removal and a matched null/shuffle/counterfactual interface.
