# Run Qwen with an existing TriviaQA retrieval JSONL

This runbook is for the GPU machine. FAISS indexing is performed elsewhere;
the FAISS index, evidence files, and embedding model do not need to be copied
to this machine.

Run all commands from the repository root.

## 1. Get the experiment branch

```bash
git clone https://github.com/Necr0x0Der/memWMLM.git
cd memWMLM
git fetch origin
git switch issue-11-triviaqa-rag
```

If the repository is already present, only fetch and switch branches.

## 2. Create the Qwen environment

Install `uv` if necessary, then create the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.12
```

This environment uses native model weights without NF4 or other quantization.

## 3. Copy the prepared files

Create the artifact directories:

```bash
mkdir -p artifacts/triviaqa/triviaqa-unfiltered
mkdir -p artifacts/triviaqa-rag
```

Copy these three files from the machine where retrieval was prepared:

```text
unfiltered-web-dev.json
dev-faiss-top5.jsonl
dev-faiss-top5.jsonl.meta.json
```

Place them at exactly these paths:

```text
artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json
artifacts/triviaqa-rag/dev-faiss-top5.jsonl
artifacts/triviaqa-rag/dev-faiss-top5.jsonl.meta.json
```

The `.meta.json` file is created next to the retrieval JSONL by
`rag_for_triviaqa.py` and is required by the Qwen runner. The FAISS directory
itself is not required.

Verify the files:

```bash
test -f artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json
test -f artifacts/triviaqa-rag/dev-faiss-top5.jsonl
test -f artifacts/triviaqa-rag/dev-faiss-top5.jsonl.meta.json
wc -l artifacts/triviaqa-rag/dev-faiss-top5.jsonl
```

The retrieval JSONL should contain 11,313 lines.

## 4. Download Qwen3-1.7B

```bash
.venv/bin/hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

## 5. Select a free GPU and run a smoke test

```bash
nvidia-smi
```

Replace `0` below with the physical number of a free GPU. Within the process,
that selected GPU is exposed as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/issue-11-triviaqa-rag/run_qwen_rag.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --retrieval-cache artifacts/triviaqa-rag/dev-faiss-top5.jsonl \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-smoke.jsonl \
  --max-examples 20 --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

Check the smoke-test summary:

```bash
cat runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-smoke.jsonl.summary.json
```

## 6. Run all 11,313 questions

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/issue-11-triviaqa-rag/run_qwen_rag.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --retrieval-cache artifacts/triviaqa-rag/dev-faiss-top5.jsonl \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-dev.jsonl \
  --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

The full command is resumable when repeated unchanged. Final files are:

```text
runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-dev.jsonl
runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-dev.jsonl.meta.json
runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-dev.jsonl.summary.json
```

Inspect the result with:

```bash
cat runs/issue-11-triviaqa-rag/qwen3-1.7b-faiss-top5-dev.jsonl.summary.json
```

The closed-book references are 16.72% EM for Qwen3-1.7B and 30.39% EM for
Qwen3-4B. Partial success is at least 23.56% EM; strong success is at least
27.39% EM.
