# Issue #5: Qwen3 factual-knowledge comparison

GitHub issue: https://github.com/Necr0x0Der/memWMLM/issues/5

## Status and scope

This directory contains the prospective protocol and later artifacts for the
Qwen3 model-size comparison. The first stage is a TriviaQA infrastructure pilot.
It is not sufficient by itself to establish a cross-dataset or cross-domain
conclusion for issue #5.

This protocol was written before inspecting any test results.

Initial local tooling:

- `RUNBOOK.md` gives the exact cross-machine execution sequence;
- `RESULTS.md` reports the completed 1.7B versus 4B TriviaQA comparison;
- `comparison.json` contains the same aggregate and paired metrics for tooling;
- `triviaqa.py` loads TriviaQA v1.0 and implements official-style EM/F1;
- `score_predictions.py` scores a JSON object keyed by `QuestionId`.
- `run_qwen.py` performs resumable deterministic Qwen inference to JSONL;
- `mmlu.py` loads official MMLU CSVs and reports per-subject/domain accuracy;
- `run_mmlu.py` performs resumable five-shot candidate-likelihood evaluation;
- `triviaqa-manifest.json` records the official archive and `dev` split hashes.
- `model-revisions.json` pins the model revisions used by this experiment.

## Pilot claim

On open-domain factual questions from TriviaQA, Qwen3-4B has higher normalized
exact-match accuracy than Qwen3-1.7B under an identical no-context, no-RAG,
no-thinking protocol.

The magnitude and error distribution will be reported without treating this
single dataset as evidence of uniform or domain-specific gains.

## Data

- Use the official TriviaQA v1.0 unfiltered `dev` split.
- Give the model the question only; do not provide evidence documents.
- Preserve the official answer value and aliases.
- Record the source URL, archive and file SHA-256 hashes, split, example count,
  and deterministic example identifiers and order.

## Models and prompt

- Compare `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`.
- Pin and report immutable Hugging Face model and tokenizer revision SHAs.
- Use the identical chat prompt with `enable_thinking=False`.
- Ask for only the shortest answer, without explanation.
- Use deterministic greedy decoding with at most 32 new tokens and record all
  generation settings.
- Do not use training, RAG, memory, a retrieval corpus, or test-time adaptation.

## Evaluation

- Use TriviaQA-style answer normalization and aliases.
- Primary metric: normalized exact match.
- Secondary metric: token F1.
- Save paired per-example predictions and formatting failures.
- Report aggregate metrics for each model, the paired difference, and a paired
  bootstrap 95% confidence interval.
- Preserve a small deterministic qualitative error set selected by explicit
  categories rather than favorability.

## Run plan and compute budget

1. Validate loaders, normalization, prompting, and output recording with local
   fixtures and tests.
2. Run a 20-example smoke test only to detect runtime or formatting failures. Do
   not use its accuracy to change the protocol.
3. Measure wall time and peak VRAM for both models without using observed quality
   to choose the sample size.
4. Target the complete official `dev` split. Make evaluation resumable so it can
   run in chunks.
5. If a full run is infeasible, amend this protocol before inspecting aggregate
   quality. Record measured throughput, the time budget, deterministic sampling
   rule, selected sample size, and minimum effect the reduced sample can resolve.

The local NVIDIA GeForce RTX 2060 with 6 GB VRAM is sufficient only for the
1.7B smoke test. Run the matched experiment on a machine that fits both models
without quantization or CPU offload. Use the models' native BF16 weights for both
sizes and record peak VRAM.

## Reproducibility record

Record all of the following with every reported run:

- repository commit SHA and exact config;
- Python, PyTorch, and Transformers versions;
- GPU, dtype, quantization, and offload settings;
- model and tokenizer identifiers and immutable revisions;
- dataset source, split, manifest, and hashes;
- prompt version and full rendered prompt contract;
- decoding settings;
- evaluation script version;
- wall time, peak VRAM, and processed example count;
- whether any evaluation data influenced implementation or parameter choices.

## Limitations

TriviaQA may have appeared in model pretraining. Qwen does not publish enough
corpus-level provenance to rule this out. This pilot therefore measures benchmark
recall and performance, not uncontaminated acquisition of factual knowledge.

One dataset cannot test whether gains are uniform across factual domains. Results
from this pilot must not close issue #5 or support the broader modular-memory
interpretation until at least one additional factual dataset and an appropriate
reasoning or broad-domain control are evaluated under a matched protocol.

## Amendments

Record any prospective protocol changes here, with a date and reason, before
inspecting the affected aggregate results.

- 2026-11-08: After completing TriviaQA, add the full MMLU test split as a
  broad-domain control. Use the official five-shot prompt, identical chat
  formatting with thinking disabled for both models, and summed conditional
  log likelihood over the answer letters A/B/C/D. Report overall, per-domain,
  and per-subject accuracy. This amendment was recorded before inspecting any
  MMLU results.
