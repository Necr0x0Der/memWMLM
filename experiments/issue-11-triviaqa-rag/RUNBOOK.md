# Prepare TriviaQA retrieval and run Qwen

FAISS indexing can be performed on a high-RAM machine and the resulting
retrieval JSONL copied to the GPU machine. The FAISS index, evidence files, and
embedding model do not need to be copied after retrieval is complete.

Run all commands from the repository root.

## 1. Get the experiment branch

```bash
git clone https://github.com/Necr0x0Der/memWMLM.git
cd memWMLM
git fetch origin
git switch issue-11-triviaqa-rag
```

If the repository is already present, only fetch and switch branches.

## 2. Prepare retrieval

Skip this section if the selected retrieval JSONL and its `.meta.json` have
already been created.

The oracle route does not require FAISS or high RAM. The optional global FAISS
route retains all passage texts in RAM; the 2.5 GB evidence archive expands to
approximately 7.3 GB, so 64 GB RAM is recommended for that route. A machine
with 16 GB RAM is insufficient for the current FAISS implementation even with
document batching.

### 2.1 Install the RAG dependencies

Install `uv` if necessary, then create the environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra rag --python 3.12
```

The `rag` extra installs LangChain, `langchain-huggingface`,
`sentence-transformers`, and `faiss-cpu`. Verify them with:

```bash
.venv/bin/python -c "import faiss, sentence_transformers; from langchain_huggingface import HuggingFaceEmbeddings; print('RAG dependencies OK')"
```

### 2.2 Download TriviaQA

Download the unfiltered question archive and extract the dev JSON:

```bash
mkdir -p artifacts/triviaqa
curl -fL --retry 3 --continue-at - \
  https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz \
  -o artifacts/triviaqa/triviaqa-unfiltered.tar.gz
tar -xzf artifacts/triviaqa/triviaqa-unfiltered.tar.gz \
  -C artifacts/triviaqa \
  triviaqa-unfiltered/unfiltered-web-dev.json
```

Download the approximately 2.5 GB TriviaQA RC archive containing Web and
Wikipedia evidence, then extract it. Allow at least 12 GB free disk space for
the archive and extraction, and preferably 20 GB including the generated index:

```bash
mkdir -p artifacts/triviaqa-rc
curl -fL --retry 3 --continue-at - \
  https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz \
  -o artifacts/triviaqa-rc/triviaqa-rc.tar.gz
tar -xzf artifacts/triviaqa-rc/triviaqa-rc.tar.gz \
  -C artifacts/triviaqa-rc
```

Verify the expected paths:

```bash
test -f artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json
test -d artifacts/triviaqa-rc/evidence/web
test -d artifacts/triviaqa-rc/evidence/wikipedia
du -sh artifacts/triviaqa-rc/triviaqa-rc.tar.gz artifacts/triviaqa-rc/evidence
```

### 2.3 Create answer-containing oracle context without FAISS

This condition tests whether Qwen can use explicitly provided knowledge. It
uses `NormalizedAliases` to select answer-containing chunks, so it is a reader
upper bound rather than a retrieval-quality measurement. It does not build a
FAISS index and does not require high RAM.

```bash
mkdir -p artifacts/triviaqa-rag

.venv/bin/python experiments/issue-11-triviaqa-rag/prepare_oracle_context.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --evidence-path artifacts/triviaqa-rc/evidence \
  --output artifacts/triviaqa-rag/dev-oracle-top5.jsonl \
  --chunk-words 300 --overlap-words 50 --top-k 5

wc -l artifacts/triviaqa-rag/dev-oracle-top5.jsonl
cat artifacts/triviaqa-rag/dev-oracle-top5.jsonl.meta.json
```

The expected count is 11,313. The metadata reports how many questions had an
answer-containing chunk and how many used ordinary associated-evidence
fallback. Never describe this condition as answer-independent RAG.

### 2.4 Optional: pin the embedding model for global FAISS

Download the embedding model and print its immutable revision:

```bash
.venv/bin/hf download sentence-transformers/all-MiniLM-L6-v2
.venv/bin/python -c "from huggingface_hub import model_info; print(model_info('sentence-transformers/all-MiniLM-L6-v2').sha)"
```

Copy the printed SHA into `<EMBEDDING_SHA>` in the next command.

### 2.5 Optional: build FAISS and create the retrieval JSONL

```bash
mkdir -p artifacts/triviaqa-rag

.venv/bin/python experiments/issue-11-triviaqa-rag/rag_for_triviaqa.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --evidence-path artifacts/triviaqa-rc/evidence \
  --index-path artifacts/triviaqa-rag/dev-evidence-faiss \
  --output artifacts/triviaqa-rag/dev-faiss-top5.jsonl \
  --embedding-revision <EMBEDDING_SHA> \
  --document-batch-size 16 \
  --top-k 5
```

Indexing prints `Indexed N/96407 documents`. After indexing finishes, retrieval
creates one JSONL record per question. Verify the result:

```bash
wc -l artifacts/triviaqa-rag/dev-faiss-top5.jsonl
test -f artifacts/triviaqa-rag/dev-faiss-top5.jsonl.meta.json
```

The expected count is 11,313. Keep the JSONL and `.meta.json` together.

## 3. Create the Qwen environment on the GPU machine

Install `uv` if necessary, then create the environment without the RAG extra:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.12
```

This environment uses native model weights without NF4 or other quantization.

## 4. Copy the prepared files to the GPU machine

Create the artifact directories:

```bash
mkdir -p artifacts/triviaqa/triviaqa-unfiltered
mkdir -p artifacts/triviaqa-rag
```

Copy the dev JSON and one prepared cache pair from the retrieval machine. For
the oracle condition, use:

```text
unfiltered-web-dev.json
dev-oracle-top5.jsonl
dev-oracle-top5.jsonl.meta.json
```

Place them at exactly these paths:

```text
artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json
artifacts/triviaqa-rag/dev-oracle-top5.jsonl
artifacts/triviaqa-rag/dev-oracle-top5.jsonl.meta.json
```

For global FAISS instead, copy the corresponding `dev-faiss-top5.jsonl` pair.
The `.meta.json` file is created next to the retrieval JSONL by its preparation
script and is required by the Qwen runner. The FAISS directory itself is not
required.

Verify the files:

```bash
test -f artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json
test -f artifacts/triviaqa-rag/dev-oracle-top5.jsonl
test -f artifacts/triviaqa-rag/dev-oracle-top5.jsonl.meta.json
wc -l artifacts/triviaqa-rag/dev-oracle-top5.jsonl
```

The retrieval JSONL should contain 11,313 lines.

## 5. Download Qwen3-1.7B

```bash
.venv/bin/hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

## 6. Select a free GPU and run a smoke test

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

## 7. Run all 11,313 questions

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

## 8. Run the oracle answer-containing condition

Use the same Qwen configuration, changing only the retrieval cache and output
name. Run a smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/issue-11-triviaqa-rag/run_qwen_rag.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --retrieval-cache artifacts/triviaqa-rag/dev-oracle-top5.jsonl \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-11-triviaqa-rag/qwen3-1.7b-oracle-top5-smoke.jsonl \
  --max-examples 20 --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```

Then run the full dev split:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/issue-11-triviaqa-rag/run_qwen_rag.py \
  --dataset artifacts/triviaqa/triviaqa-unfiltered/unfiltered-web-dev.json \
  --retrieval-cache artifacts/triviaqa-rag/dev-oracle-top5.jsonl \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --output runs/issue-11-triviaqa-rag/qwen3-1.7b-oracle-top5-dev.jsonl \
  --batch-size 1 --max-new-tokens 32 \
  --device cuda --dtype bfloat16
```
