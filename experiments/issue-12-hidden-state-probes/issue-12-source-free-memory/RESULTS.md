# Source-free memory: results

## Completed pilot — reviewed 2026-11-28

Artifacts: `artifacts/pilot-none-shuffled/`. Validation NLL selected
`last_token`, layer 28, epoch 5, with frozen Qwen3-4B and a four-slot adapter.

| Metric | Memory | Question only |
|---|---:|---:|
| Mean balanced accuracy | 55.06% | 47.62% |
| All seven answers correct | 4/96 | 0/96 |
| Invalid answers | 0/672 | 0/672 |

**The source-free interface gives a limited positive signal (+7.44 percentage
points), but reliable use of a coherent spatial state was not demonstrated.**
Memory answers contain at least one contradictory inverse-relation pair in
82/96 situations. Formatting failures do not explain the weak performance.

For Direction C, a saved situation vector delivered through a learned adapter
can improve answers without the source text. This does not establish a prepared
or causally used world state: the adapter has 26.2 million trainable parameters,
memory-swap/null/oracle controls are absent, and only four independent test
layouts are available. Representation, interface, and training limitations
remain possible explanations for the weak result.

Both sets of 672 answers were checked against dataset labels; recomputed metrics
match the report. Dataset and checkpoint hashes were verified. Feature extraction
and model inference were not rerun; the source feature cache was unavailable.

## Historical status as of 2026-11-26

**No QA results with a trained Qwen3-4B adapter were available at this stage.**

A separate source-free soft-prefix experiment was implemented; protocol and
commands are in [README.md](README.md). Memory swapping was deferred at the
user's request.

The attempt on the local machine stopped before training or test evaluation:

```
RuntimeError: CUDA is unavailable; no training or test evaluation was run.
```

Environment checks at the time:

- `torch.cuda.is_available()` returned `False`;
- `/dev/nvidia*` was absent, and `nvidia-smi` could not connect to the driver;
- the standard HF cache contained Qwen3-1.7B but not Qwen3-4B;
- the Qwen3-4B feature cache referenced by earlier reports was unavailable locally;
- PyTorch 2.15.0+cu132, Transformers 5.22.1.

1.7B was not substituted for 4B, as that would change the model under study.
A CPU run with a small randomly initialized Qwen checks implementation
correctness, not the usability of the real model's representations.

Four new tests passed, including a complete two-epoch training cycle, validation
selection, generation, and report writing with a small random Qwen. Checks
covered frozen backbone weights, adapter gradients, absence of source/target
in the question prompt, masking, and correct greedy generation with `inputs_embeds`.

Continuing requires an available GPU, pinned Qwen3-4B, and a complete feature
cache (or fresh extraction using the README command). After training, inspect
`test-memory.jsonl` and `test-question_only.jsonl` first, then `report.json`.
Do not select a new pooling or layer based on those test answers.

At the time of this status entry, issue #12 provided context and the new protocol
had only been recorded locally; no external comments, commits, or pushes had
been made for this experiment.
