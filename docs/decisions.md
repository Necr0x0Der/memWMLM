# Architecture decisions

Concise record of decisions that govern experiments in this repository.

## 2026-10-09 — Axial backbone family

Start with dense Qwen3, not Qwen3.5 or later hybrid backbones.

- First end-to-end development backbone: `Qwen/Qwen3-1.7B`.
- First meaningful replication: `Qwen/Qwen3-4B`.
- Rationale: Qwen3 is a conventional dense causal decoder with an easily controlled KV cache and injection points. Qwen3.5's Gated DeltaNet/full-attention hybrid introduces unrelated questions about where and how to insert memory.
- Constraint: memory code consumes/returns hidden states through an axial-model interface; do not hard-code Qwen3 attention implementation throughout the project.

## 2026-10-09 — v0 memory protocol

Use a resettable two-episode meta-learning pair:

```text
source → commit → application
```

- Runtime memory resets before every pair.
- The source is processed by normal causal attention and cannot read memory.
- The application cannot see source tokens; it can only access the committed state through a memory reader.
- Only application loss is supervised.
- Axial model weights are frozen.

This is a writer/reader plumbing sanity check, not evidence of persistent semantic memory.

## 2026-10-09 — v0 memory representation

Begin with a pool of writable embedding-memory segments, not nonlinear experts.

- Writer emits source-derived keys and values.
- Reader uses additional learned query/key/value projections to attend to segments.
- Returned features enter selected upper axial layers through a zero-initialized gated residual path.
- Nonlinear mutable experts are a later substitute behind the same writer/read interface, to be tested on multi-commit interference/composition tasks.

## 2026-10-09 — first task and axial-model control

Use bAbI Tasks v1.2, English 10k Task 1:

```text
tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{train,test}.txt
```

First establish frozen Qwen performance with source plus application in ordinary context. Use one fixed bAbI instruction, no-thinking mode, and candidate likelihood over the finite location vocabulary. Do not initially add LoRA or a task-format adapter; that would be a separately declared experiment if the frozen prompt interface proves inadequate.

## 2026-10-12 — Result: pooled embedding-memory reset-pair baseline

The first implementation used a Perceiver-style writer and late residual reader:

```text
source final Qwen states
→ 32 learned write queries attend over all source positions
→ one 512D key/value cell per query
→ application-only reader cross-attends to cells
→ gated residual injection into the final 1–4 Qwen layers
```

- bAbI Task 1 full-context frozen Qwen: about 82% accuracy; application-only: 0%.
- Pooled memory plateau: about 60% training / 55% test accuracy; candidate likelihood slightly exceeded free generation after output formatting/EOS was fixed.
- `null_memory` and `shuffled_memory` were both about 17%, so the learned runtime state is source-specific and necessary, but this does not establish an adequate episodic/world/semantic representation.
- Increasing cell count, memory width, attention-head count, or injection layers did not materially change the plateau; one injection layer was close to four. Training loss remained noisy and did not converge to a near-zero train solution.
- Separate source-side and application-side rank-8 LoRA adapters made optimisation more stable and raised both train and test by only about 2.5 percentage points. This confirms a small interface-adaptation effect but does not solve the bottleneck.

Interpretation: the limitation is structural rather than simple capacity. The writer recomputes query-agnostic global summaries; it has no sequential allocation, fact/entity slots, overwrite operation, explicit positional/event trace, or direct cell supervision. The reader also enters late through a new residual channel rather than Qwen's native layerwise attention K/V path.

**Decision:** stop hyperparameter scaling of this pooled-cell branch. Preserve it as a source-specific-channel baseline. Next implement an episodic-attention control that transfers raw source cache/layerwise K/V in Qwen's native attention format across the source/application boundary. This is an episodic-memory control, not the target semantic-memory architecture. Subsequent memory representations remain open for discussion; compression is a possible experiment, not a settled objective.

## 2026-10-13 — First KV control: frozen native continuation

Implement the first, **non-training** control: tokenize the complete full-context prompt once, prefill the native Qwen KV cache through the story, then continue with the question/answer and score candidates using that cache. Compare logits and candidate predictions with a single-pass full-context computation on exactly the same tokens, including position handling. No trainable writer, reader or adapter is involved. This verifies the processing boundary, not a learned memory interface or semantic knowledge acquisition. The later learned-memory experiments also measure task learning; frozen full-context accuracy is not an upper bound on an adapted model.

Alex's full bAbI Task 1 test: frozen `full_context` candidate accuracy 81.9%, native-KV continuation 82.4% across 1,000 examples; 15 predictions differ. These are two candidate-likelihood evaluations, not free-generation versus candidate scoring. A checked disagreement already shows source logits differ between full and short-prefix passes, consistent with numerical/kernel differences, while exact equivalence remains unproven. Treat this as approximate performance parity, not a statistically meaningful KV improvement or a semantic-memory result.

## 2026-10-13 — Learned uncompressed episodic side-path (experimental design)

Next test a separate trainable reader over all source-token per-layer Qwen K/V without pooling or compression. The source is encoded with frozen Qwen and is not present in the application context; the application starts its own positional context and may access source only through gated cross-attention side paths in the last four layers. Train reader plus optional application-side rank-8 LoRA, while keeping Qwen base/source encoder frozen. Measure candidate accuracy on train/test and shuffled/null-memory controls. Alongside it, train a frozen-base full-context rank-8 LoRA task control on the same bAbI task, answer+EOS loss, train examples/optimizer steps and upper q/o projection placement. These are not identical parameter budgets: the side reader has additional parameters and sees a separate source prompt. A result above frozen 82% can reflect task adaptation, not memory quality alone. The exact future representation beyond this episodic upper-bound probe remains open.

## 2026-10-14 — Test early and spaced injection in pooled-cell memory

Alex reports that episodic KV-reader memory reached about 33% after 2,000
single-example steps, then about 55–56% with more intensive training; it is
**not** a 56% result after only 2,000 steps. The task-adapted full-context LoRA
reached about 98% train / 94% test. These are reported runs, not evidence that
the episodic model has converged. Both learned memory variants so far inject
only at late layers. A late-one versus late-four comparison does not test
whether early memory access lets the model combine the question and source.

Add explicit zero-based `injection_indices` to the original *pooled embedding
memory* experiment; retain `injection_layers` as the old last-N default. Keep
the number of reader positions equal for late-four versus spaced-four
(`0,9,18,27` when those layers exist) and possibly early-four; compare with
the same training budget and shuffled/null-memory controls. Hooks add memory
**after** the selected block. A later run will determine whether placement,
as opposed to writer/representation or training, explains the gap.

### Reported spaced-injection ablations (2026-10-14; interpretation open)

Alex trained the pooled-cell model with injection after blocks `[0,9,18,27]`, a larger batch and longer run. With both source- and application-side LoRA, training accuracy reached ~99.99% and test accuracy ~54%. Without LoRA, training improved only slightly versus earlier runs and test did not improve. With application-side rank-8 LoRA, training reached ~77% at the compared step count and test began to decline. Source-only rank-8 LoRA rapidly fit the training set but reached ~52.5% test. The task-adapted ordinary full-context LoRA control previously reached ~98% train / ~94% test. These are user-reported measurements under differing optimization conditions, not a controlled attribution of the gain to an early reader or a proof that the model memorizes examples in weights.

Working hypotheses to distinguish next: source LoRA may make source states easier to write and increase effective task capacity, while the train–test gap may arise from memorizing training patterns instead of a reusable source/question algorithm. Test whether predictions causally depend on the *correct* runtime memory (train and test shuffled/null controls), and probe counterfactual source swaps and novel recombinations of names, facts and questions with a held-out generator split. Compare learning curves and matched budgets before selecting layer placement or regularization as the cause.

## 2026-10-15 — Early-only pooled result and episodic-reader placement test

Alex reports that pooled-cell injection after `[0,1,2,3]` achieved up to **72.3% test**, with memory width 2048; the effect of width was not isolated. More or fewer early injection blocks somewhat reduced test accuracy, but remained better than other tested placements. Spaced `[0,9,18,27]` with LoRA reached nearly 100% train yet only 50–60% test; upper-only injection had not materially exceeded ~60% even on train. Early access appears important **in these runs**. A plausible interpretation is that early-only reading forces more downstream processing of memory, whereas a late injection plus LoRA allows short-circuit fitting; this is not established as a mechanism or a universal constraint. The remaining gap from ~82% frozen full-context could reflect event-order loss in pooled memory or other interface/data effects. Do not spend further effort tuning pooled-memory hyperparameters now.

Next ablation: allow explicit post-block `injection_indices` in the separate episodic source-token K/V reader, retaining last-N defaults and checkpoint compatibility. Test early-four placement with the same source representation and training protocol as the earlier late-four episodic run. This is not native Qwen cache continuation and still trains one example per step. Compare matched exposure budgets and causal source controls before interpreting accuracy differences. Alex reports no improvement from this episodic early-layer variant on GPU.

## 2026-10-20 — Same-task training-volume probe

Alex reports that the best pooled early-layer model can reach exactly 100% train while test is ~72.3%. This is insufficient to establish an architectural test ceiling; he requests an extra 10k Task 1 training questions concatenated to the unchanged original 10k. Use a separately labeled, reproducible *approximate* QA1 generator, not a different reasoning task: same four actors, six places, movement phrases, 15-line episodes, five questions and latest-location support. The public Lua/Torch generator is itself a rewrite that does not exactly reproduce the released v1.2 distribution, cannot run on this host, and has no straightforward fixed seed; the Python generator also is not the unpublished original. Preserve the original test verbatim and label the mixed-source 20k train. Store the generated files and hash/stats manifest in `datasets/qa1-extended-20k/`; report distribution shift and train/test overlaps. Compare the existing architecture at matched settings and explicit numbers of presented examples, without treating a gain or loss from mixed-origin training as clean proof of capacity or insufficiency. Do not change the writer's question-blind protocol.

Alex reports ~77% accuracy on the unchanged published test after training with the mixed 20k train (versus up to 72.3% with 10k). To extend the same volume probe, generate 40k total train with the same seed and generator: original 10k + 30k synthetic, with the previously published 20k as an exact byte prefix. Keep the original test unchanged, store the incremental 20k and provenance under `datasets/qa1-extended-40k/`, and compare against the 20k run with the same model configuration. The increase to 77% is user-reported, not a controlled proof that the architecture will reach 82% at larger sizes.

### 2026-10-20 — 40k run: reported results and handoff

**Correction from Alex:** after training on the mixed 40k train, the model reached **99.7% accuracy on the unchanged published test**, not on train. He stopped before fully completing training; 40k-run train accuracy was not reported. The same reported evaluation gave **16.7% for `shuffled_memory`**. The earlier early-only 10k run could reach exactly 100% train and up to 72.3% published test; the 20k mixed-origin run reached ~77% published test. Thus the 40k result is a strong improvement on this held-out split, but these measurements alone do not disentangle dataset coverage, memorization in trainable weights, and a transferable memory mechanism. The shuffled control shows source-specific runtime state matters for the reported checkpoint, not *how* the learned solution generalizes.

Alex proposes discussing a new dataset in a fresh conversation because this QA1 task may be saturated, while deferring architectural changes to a later stage. This is a proposed direction, not yet a selected dataset or architecture. In the next discussion, use this file plus `topics/llm-alternatives.md` as the starting context; do not confuse the **99.7% 40k-run test** result with train accuracy.

## 2026-10-20 — First QA2 probe (user-reported)

On the original bAbI v1.2 `en-10k/qa2_two-supporting-facts` files, frozen Qwen3-1.7B with the full story and candidate-likelihood scoring reached **49.6% test**. The pooled-memory model trained on the original **10k QA2 train**, using Alex's successful QA1 settings without an intended architecture change, reached **99.4% train / 81.6% test**; Alex stopped the initial training run early. These are not matched frozen-versus-trained capacity controls: the memory writer/reader (and any configured adapters) learned the task, whereas the 49.6% full-context baseline did not. Exact run config, checkpoint step, exposure count and QA2 shuffled/null-memory results have not been recorded here; do not infer them.

QA2 requires linking possession and movement, rather than only looking up a person's latest location, and this 10k result does not saturate its test. A task-adapted full-context LoRA control was subsequently run (below); shuffled/null-memory QA2 controls are not a priority for Alex. More QA2 data remains an open option rather than ruled out.

## 2026-10-20 — QA2 full-context LoRA and QA1+QA2 mixture (user-reported)

Full-context Qwen3-1.7B with trained LoRA on the original QA2 10k scored **69.1% train / 56.1% test**. This implementation adapts q/o projections of the last four blocks, indices `[24,25,26,27]`; the source and question still share native attention. Compared with QA2 pooled early-access memory's **99.4% train / 81.6% test**, the result is consistent with the writer/reader providing useful trainable task capacity, but does not establish that LoRA is intrinsically too small: placement, optimization, parameter count and presented-example budgets are not matched/recorded here.

Alex also trained the pooled-memory model on a combined **original QA1 10k + QA2 10k** dataset. Training was harder but reached **100% train on each task**; test fell to **61% QA1 / 57% QA2** (Alex's corrected task-to-score assignment). Do not confuse this with QA1's synthetic 40k extension. Before attributing this to sequential task learning: `scripts/train_babi_memory.py` loads all examples in a single list and uses `rng.choices(examples, k=batch_size)` at every step; a literal 10k+10k concatenation of valid bAbI files is sampled uniformly with replacement across both tasks, including within a batch. This is not sequential batching. Verify the actual merged file, answer vocabulary, exact config, checkpoint, exposures and per-task evaluation paths before attributing the generalization regression to interference, overfitting or a loader bug. No mixed-training protocol change or data expansion has yet been selected.

## 2026-10-21 — Six-task evaluation and mixed checkpoint (user-reported)

Alex evaluated the pooled-memory variant with post-block injection `[0,1,2,3]` and the same hyperparameters on six original bAbI `en-10k` tasks. In the table, `Spaced` is Alex's label for **separately trained per-task checkpoints**; the placement is early-only, not `[0,9,18,27]`. Entries are train/test accuracy where both were supplied; a lone frozen-Qwen value is test accuracy. `Mixed` is one checkpoint trained across all six tasks; its reported metrics are from a checkpoint **before** a loss spike after 40,000 steps, and are not fully converged.

| Task | Frozen full-context Qwen | Per-task memory (`Spaced`) | Six-task memory (`Mixed`) |
| --- | --- | --- | --- |
| `qa1_single-supporting-fact` | test 0.82 | 1.0 / 0.723 | 0.7733 / 0.572 |
| `qa2_two-supporting-facts` | test 0.496 | 0.994 / 0.816 | 0.9181 / 0.718 |
| `qa16_basic-induction` | 0.768 / 0.778 | 1.0 / 0.522 | 0.752 / 0.477 |
| `qa4_two-arg-relations` | 0.3496 / 0.341 | 1.0 / 0.994 | 0.989 / 0.979 |
| `qa12_conjunction` | 0.665 / 0.664 | 0.997 / 0.698 | 0.8032 / 0.708 |
| `qa11_basic-coreference` | 0.6802 / 0.68 | 1.00 / 0.814 | 0.8697 / 0.759 |

Per-task memory beats frozen full-context Qwen on **four of six tests** (QA2, QA4, QA12, QA11), but not QA1 or QA16. Mixed is lower than the corresponding per-task checkpoint on **five of six tests**; QA12 improves slightly (0.698 → 0.708). Mixed is above frozen full-context on QA2, QA4, QA12 and QA11, and below on QA1 and QA16. The QA1 40k synthetic-extension result (99.7% unchanged test) is a *different* per-task training regime from this 10k comparison.

The six-task run learned more slowly; Alex reports a sharp loss increase after 40,000 steps that did not recover, so only a pre-degradation checkpoint was evaluated. This does not yet establish lack of a task-general memory mechanism: joint optimization may be incomplete/unstable, and the evaluation uses a shared prompt that asks for a "location word" even on QA16, whose answer vocabulary contains colors. Check exact mixed data construction, per-task answer vocabularies and evaluation paths, learning-rate/gradient-norm trajectories and checkpoint provenance before attributing the gap to architecture. Alex plans to test training settings; his proposed milestone is a *single* six-task memory checkpoint at least matching frozen full-context Qwen on each test. No architecture change has been selected.

## 2026-10-21 — Prompt versions for six-task comparison

The **historical six-task table above uses the legacy application instruction, exactly**: `This is a bAbI question-answering task. Answer with exactly one location word from the story. Do not explain your answer.` The source instruction remains `Read and retain the following story for a later question.` The word `location` is mismatched to QA16 (colors). These recorded measurements must retain their prompt provenance and not be silently relabeled after code changes.

The alternative `general_one_word` application instruction is: `This is a bAbI question-answering task. Answer with exactly one word. Do not explain your answer.` It is now the **default for new** frozen full-context Qwen evaluations, memory training, and full-context LoRA training; pass `--prompt-version legacy_location` to reproduce historical frozen evaluations or train a new model under the old instruction. New checkpoints record the prompt version, and memory/LoRA evaluation defaults to the checkpoint's version (old checkpoints without metadata mean `legacy_location`). An explicit conflicting version in evaluation is rejected. A change of instruction creates a **new experiment**: compare frozen full-context and newly trained memory with the same new instruction, same split/scoring protocol, while leaving the old table intact. The original fixed source instruction, tasks and labels are unchanged.

## 2026-10-22 — Completed lower-LR six-task run (user-reported)

Alex completed the six-task pooled-memory training with a lower learning rate, avoiding the earlier irreversible loss spike. Train accuracy reached near 100% across tasks, while test results mostly worsened versus earlier checkpoints; exact per-task test scores, learning rate, checkpoint steps and prompt version for this run were not supplied. This is a train–test overfitting pattern, not evidence that joint-training convergence alone closes the gap. In the separate QA1 volume probe, adding *same-task* examples had raised published-test accuracy nearly to 100%; adding related-task examples in joint training has not helped so far. Alex plans to try reducing capacity. This leaves open whether the writer/reader learns broadly reusable memory semantics, task-specific memory coding, or a trainable aid for Qwen's task-solving computation. A two-arbitrary-prompt inference probe is requested to explore behavior beyond bAbI; it will be exploratory, not a calibrated general-language benchmark.
