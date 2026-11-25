# Shared attention pooling + linear probes (Qwen3-4B)

This is a separate follow-up to the fixed-pooling results. For each of the eight
instruction/order conditions, train **one shared query q and seven linear heads**.
Every head sees the same pooled vector z. The Qwen backbone remains frozen,
receives only the situation (with its instruction), and never receives a question.
The query and heads learn jointly from the seven existing dataset labels.

## Model and selection

For token states h at one layer, apply train-only feature normalization,
`x = (h - mean) / scale`, and compute:

```text
weights = softmax(x @ q / sqrt(hidden_size))  # padding masked out
z = sum(weights * x)
logits[question] = W[question] @ z + bias[question]
```

There are no key/value projections, extra attention layers or MLP. All valid
input tokens, including instruction tokens, participate. q starts at zero,
so its initial pooling is uniform. Each situation has its own weights and z;
q is common across situations and questions within a condition. The bank has
its own q for each instruction/order condition.

Loss is the mean of seven cross-entropies, plus L2 on q and the mean squared
head-weight norm; biases are unpenalized. Adam jointly updates q and heads.
Means and scales are computed from per-situation mean vectors on train only
and applied to every token. This makes q=0 equivalent to a standardized
mean-pooled input. The shared query is label-supervised, not next-token-trained
or task-independent in the strong sense. The combined readout is nonlinear.

**Selection changes relative to the earlier per-question experiment:** because
all heads must read the same z, layer, L2 and epoch are selected **for the whole
bank**, using mean validation balanced accuracy across questions. Ties prefer
joint exact-target accuracy, stronger L2, earlier epoch, then first layer.
The layer cannot be individually selected per head while preserving this
single-layer shared-z architecture. Test states are not loaded until all banks'
selection is complete; no train+validation refit is performed.

## 1. Preserve token states

The earlier extractor saved only last-token and pooled vectors. Attention
cannot recover individual token states from those files. The new optional
`--save-token-states` flag preserves them during the **same frozen forward pass**;
existing defaults and pooled outputs remain available. Re-extraction is required
unless a complete compatible per-token cache already exists.

From repository root, with the existing environment, dataset and cached model:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  experiments/issue-12-hidden-state-probes/extract_hidden_features.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/token-features-qwen3-4b \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --layers 4,8,12,16,20,24,28,last \
  --save-token-states \
  --device cuda --model-dtype bfloat16 --output-dtype float16 \
  --batch-size 4 --max-length 256
```

Each shard now also contains `token_states`, a list of tensors shaped
`[selected_layers, valid_tokens, hidden_size]`, in the same order as `ids`.
Padding is removed before saving. The feature manifest records this capability.
Original token order and instruction tokens are retained.

Storage is substantially larger: approximately
`number_of_rows × average_valid_tokens × number_of_layers × hidden_size × 2`
bytes for FP16 token states alone. For example, 4,608 rows × 100 tokens × 8 layers
× 2,560 dimensions requires about 18.9 GB (17.6 GiB), excluding pooled outputs
and serialization overhead. Actual length depends on tokenization. Training
loads one layer at a time on CPU, batches it onto the selected device, and
re-reads shards for different layers; allow RAM and disk bandwidth accordingly.
Use a new output directory for any rerun; do not overwrite existing artifacts.

## 2. Train the attention banks

```bash
.venv/bin/python experiments/issue-12-hidden-state-probes/probe_training/train_attention.py \
  --dataset-dir experiments/issue-12-hidden-state-probes/artifacts/db \
  --feature-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/token-features-qwen3-4b \
  --output-dir experiments/issue-12-hidden-state-probes/probe_training/artifacts/attention-qwen3-4b \
  --mode attention --layers all --l2-values 0.0001,0.001 \
  --epochs 50 --lr 0.001 --batch-size 16 \
  --device cpu --threads 1 --seed 12012 --bootstrap-samples 2000
```

For GPU training, use `CUDA_VISIBLE_DEVICES=0` and `--device cuda`. Qwen is not
loaded during training. `--layers 8,16` can restrict the candidate grid; choose
the grid before inspecting test performance. Epochs are validation candidates,
not repetitions selected on test. With only three validation groups, these
choices remain exploratory. A single seed does not quantify training variance.

## 3. Run a matched mean-pooling control

Repeat the training command with `--mode mean` and a different output directory,
for example `artifacts/attention-mean-control-qwen3-4b`. All other settings should
match. In this mode q stays zero and is not optimized. The control uses the same
heads, normalization, optimizer, shared layer/L2/epoch selection and token inputs.
This isolates learned pooling more closely than comparing only with the earlier
LBFGS experiment with independently selected per-question layers.

Both modes also retrain q/heads (q fixed in mean mode) on train group-shuffled
labels at the selected layer/L2/epoch. This is a fixed-configuration sanity check,
not a permutation significance test or a causal state intervention.

## 4. Outputs and interpretation

The new output directory contains:

- Eight `instruction--order.pt` checkpoints, each holding q, normalization,
  seven heads, class definitions, condition and selected configuration.
- `report.json`: configuration/provenance, validation scores for every candidate
  epoch, per-question and joint metrics, shuffled-label control, paired
  instruction contrasts and Mary-selectivity differences.
- `test_predictions.jsonl`: one row per test input, with all predicted classes.

The report retains the same metric/contrast meanings as [RUNBOOK.md](RUNBOOK.md).
A test improvement demonstrates that this learned readout extracts more useful
information than the control. It does not prove the frozen LLM formed a prepared
world state, or can use z without source tokens. Attention reads the full token
trace, and may learn surface shortcuts. Full per-token caches retain contextual
order information even though the final weighted sum is permutation invariant
with respect to reordering an already computed set of vectors.

For programmatic inspection, `AttentionProbeBank.pool(tokens, mask)` returns
both z and token weights. Checkpoints store all parameters needed to recreate
the bank. Attention weights alone are not a causal explanation.

Failed runs may leave partial output checkpoints; no resume is implemented.
Use a fresh output path. The final report is written only after completion.
No full Qwen extraction/training result is claimed by adding these scripts.
