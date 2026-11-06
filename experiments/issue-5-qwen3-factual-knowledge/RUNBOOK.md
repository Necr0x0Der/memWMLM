# TriviaQA Qwen3 experiment runbook

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
