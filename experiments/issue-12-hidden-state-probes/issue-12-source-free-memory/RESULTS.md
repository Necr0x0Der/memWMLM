# Status as of 2026-11-26

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
