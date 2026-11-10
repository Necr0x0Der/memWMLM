# Qwen3 factual-knowledge experiment runbook

Run commands from the repository root. Use the same checked-out commit for both
models. The final comparison uses the native BF16 weights for both sizes, without
quantization or CPU offload. The earlier local 1.7B FP16 run is only an
infrastructure smoke test.

## 1. Install uv and prepare the environment

The machine needs Linux, Git, `curl`, a BF16-capable NVIDIA GPU, a working NVIDIA
driver, and enough VRAM to hold Qwen3-4B plus inference activations. A system
Python installation is not required. At least 12 GB VRAM is recommended; 16 GB
or more leaves safer headroom.

Install the pinned `uv` release with Astral's official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
```

Start a new shell so the installed executable is on `PATH`, then install Python
and create the repository-local `.venv`:

```bash
uv --version
uv python install 3.13
nvidia-smi
uv sync --extra dev --python 3.13
```

Verify the isolated environment and CUDA before downloading models:

```bash
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
.venv/bin/pytest -q tests/test_triviaqa_pilot.py
```

Do not continue if CUDA or the tests fail.

## 2. Download and verify TriviaQA

Only the unfiltered `dev` file is required.

```bash
mkdir -p artifacts/triviaqa
curl -fL --retry 3 --continue-at - \
  https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz \
  -o artifacts/triviaqa/triviaqa-unfiltered.tar.gz
echo "7f0c94207fd0a1eb7bba3d52cdfe0735c98de850b9e5aa4e75f2776ef9a531d0  artifacts/triviaqa/triviaqa-unfiltered.tar.gz" | sha256sum -c -
tar -xzf artifacts/triviaqa/triviaqa-unfiltered.tar.gz \
  -C artifacts/triviaqa \
  triviaqa-unfiltered/unfiltered-web-dev.json
echo "720e25ee87c04444f73ab1cadad9d8d9fd3449c38e739a1a4f4ae2484f2b2009  artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json" | sha256sum -c -
```

The parser must report 11,313 examples:

```bash
.venv/bin/python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/issue-5-qwen3-factual-knowledge'); from triviaqa import load_triviaqa; print(len(load_triviaqa(Path('artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json'))))"
```

## 3. Download the pinned models

```bash
.venv/bin/hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
.venv/bin/hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c
```

The inference script uses local files only and will fail rather than silently
switching model revisions.

## 4. Run Qwen3-1.7B

First run the 20-example smoke test:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_qwen.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/qwen3-1.7b-bf16-smoke.jsonl \
  --max-examples 20 --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

If it completes without empty or malformed records, run the complete dev split:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_qwen.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/qwen3-1.7b-bf16-dev.jsonl \
  --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

## 5. Run Qwen3-4B

Use exactly the same settings and example order.

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_qwen.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/qwen3-4b-bf16-smoke.jsonl \
  --max-examples 20 --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

After a successful smoke test:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_qwen.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/qwen3-4b-bf16-dev.jsonl \
  --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

## 6. Resume and preserve outputs

The script appends one complete JSON record at a time. To resume an interrupted
run, execute the exact same command with the same output path. It skips existing
question IDs and rejects changed run metadata.

Preserve these files for each run:

- the `.jsonl` predictions;
- the adjacent `.jsonl.meta.json` metadata;
- terminal output containing wall time and peak VRAM;
- the repository commit SHA.

Do not inspect aggregate quality between the two smoke tests. Smoke runs are for
runtime and formatting validation only. Do not change prompts, decoding, example
order, precision, or quantization between the two full runs.

## 7. Download and verify MMLU

MMLU is a second, broad-domain control. Download the immutable `cais/mmlu`
archive revision and verify its published SHA-256 before extraction:

```bash
mkdir -p artifacts/mmlu
curl -fL --retry 3 --continue-at - \
  https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/data.tar \
  -o artifacts/mmlu/data.tar
echo "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b  artifacts/mmlu/data.tar" | sha256sum -c -
tar -xf artifacts/mmlu/data.tar -C artifacts/mmlu
```

Verify that all 57 subjects and 14,042 test questions are present:

```bash
.venv/bin/python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/issue-5-qwen3-factual-knowledge'); from mmlu import load_mmlu; x=load_mmlu(Path('artifacts/mmlu/data')); assert len(x)==14042; assert len({e.subject for e in x})==57; print(len(x), len({e.subject for e in x}))"
.venv/bin/pytest -q tests/test_mmlu.py
```

## 8. Run MMLU on Qwen3-1.7B, then Qwen3-4B

The smoke test evaluates one test question from every subject (57 total). It
uses each subject's five official dev examples as demonstrations. Candidate
likelihood scoring selects A/B/C/D without generating or parsing free text.

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu/data \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-qwen3-1.7b-bf16-smoke.jsonl \
  --max-examples-per-subject 1 --batch-size 1 \
  --device cuda --dtype bfloat16

.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu/data \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-qwen3-1.7b-bf16-test.jsonl \
  --batch-size 1 --device cuda --dtype bfloat16
```

After both 1.7B commands succeed, run the same smoke/full sequence for 4B:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu/data \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-qwen3-4b-bf16-smoke.jsonl \
  --max-examples-per-subject 1 --batch-size 1 \
  --device cuda --dtype bfloat16

.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu/data \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-qwen3-4b-bf16-test.jsonl \
  --batch-size 1 --device cuda --dtype bfloat16
```

Each run writes predictions, immutable metadata, and an adjacent
`.summary.json` containing overall, per-domain, and per-subject accuracy. Runs
resume safely when the identical command and output path are reused. Preserve
all three files, plus terminal timing and peak-VRAM output. Do not inspect the
1.7B full accuracy before starting the matched 4B run, and do not alter the
prompt, five-shot examples, scoring, precision, or batch semantics between
models.

## 9. Prepare one deterministic shuffled-choice MMLU control

This sensitivity control uses the same MMLU questions and five-shot
demonstrations. It changes only the order of the four choices in each test
question and remaps the correct A/B/C/D label. The seed and per-example mapping
are saved in `permutation-manifest.json`. The generated dataset remains under
`artifacts/` and must not be committed.

Do not create the output directory beforehand. Run:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/prepare_mmlu_shuffled.py \
  --source-dir artifacts/mmlu/data \
  --output-dir artifacts/mmlu-shuffled-seed-20261109/data \
  --seed 20261109
```

Verify that all questions remain present and that the manifest records the
same seed:

```bash
.venv/bin/python -c "import json,sys; from pathlib import Path; sys.path.insert(0, 'experiments/issue-5-qwen3-factual-knowledge'); from mmlu import load_mmlu; p=Path('artifacts/mmlu-shuffled-seed-20261109/data'); x=load_mmlu(p); m=json.loads((p/'permutation-manifest.json').read_text()); assert len(x)==14042 and m['example_count']==14042 and m['seed']==20261109; print(len(x), m['identity_permutation_count'])"
.venv/bin/pytest -q tests/test_mmlu.py
```

The identity permutation is one of the 24 valid random orders, so a small
number of questions will retain their original order. Do not regenerate the
dataset with a different seed after inspecting results.

## 10. Run the shuffled-choice control on both models

First run the matched 57-question smoke tests:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu-shuffled-seed-20261109/data \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-shuffled-seed-20261109-qwen3-1.7b-bf16-smoke.jsonl \
  --max-examples-per-subject 1 --batch-size 1 \
  --device cuda --dtype bfloat16

.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu-shuffled-seed-20261109/data \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-shuffled-seed-20261109-qwen3-4b-bf16-smoke.jsonl \
  --max-examples-per-subject 1 --batch-size 1 \
  --device cuda --dtype bfloat16
```

After both smoke tests pass, run the two complete 14,042-question evaluations:

```bash
.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu-shuffled-seed-20261109/data \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-shuffled-seed-20261109-qwen3-1.7b-bf16-test.jsonl \
  --batch-size 1 --device cuda --dtype bfloat16

.venv/bin/python experiments/issue-5-qwen3-factual-knowledge/run_mmlu.py \
  --data-dir artifacts/mmlu-shuffled-seed-20261109/data \
  --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output runs/issue-5-qwen3-factual-knowledge/mmlu-shuffled-seed-20261109-qwen3-4b-bf16-test.jsonl \
  --batch-size 1 --device cuda --dtype bfloat16
```

Use the same generated data directory for both models. The runner includes the
permutation-manifest hash and every generated CSV hash in each run's metadata.
As before, rerunning an identical command resumes an interrupted output safely.
