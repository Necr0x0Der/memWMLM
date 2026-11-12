# Issue #11: provided-evidence TriviaQA RAG

GitHub issue: https://github.com/Necr0x0Der/memWMLM/issues/11

This experiment asks whether Qwen3-1.7B can use provided factual knowledge to
close part of its closed-book TriviaQA gap to Qwen3-4B. It is an exploratory
RAG/context baseline, not a test of the native memWMLM sidecar.

## What exactly was done (oracle context)

- **Corpus:** the official TriviaQA RC archive (`triviaqa-rc.tar.gz`, about
  2.5 GB), containing Web search-result files and Wikipedia page files.
- **Association:** each question in `unfiltered-web-dev.json` lists associated
  `SearchResults` (Web files) and `EntityPages` (Wikipedia files).
- **Chunks:** every associated file is split into 300-word passages with
  50-word overlap using `RecursiveCharacterTextSplitter`.
- **Gold aliases:** `Answer.NormalizedAliases` supplies the accepted answer
  variants, for example `scorpio` for “What star sign is Jamie Lee Curtis?”.
- **Oracle selection:** for each question, `prepare_oracle_context.py` examines
  chunks from that question's associated files and selects up to `top_k=5`
  chunks containing a gold alias. If none contains an alias, it falls back to
  the first five associated chunks.
- **Supplied knowledge:** the selected chunk text is passed to Qwen as context;
  the model must produce the short answer from that context.

The experiment infrastructure supplies these chunks and uses the gold answer
to choose them. This makes the condition an oracle reader upper bound. It
measures how much useful evidence the model can consume when answer-containing
context is already available; it does not measure answer-independent retrieval
quality.

Issue #5 established the closed-book 1.7B versus 4B factual gap and raised the
question of whether external factual context could substitute for model size.
Issue #11 initially proposed global FAISS retrieval as the delivery mechanism.
That design would mix two effects: whether useful evidence was retrieved and
whether each model could use the evidence. The experiment was therefore
refined to oracle selection and run for both model sizes with the same cache.
This directly measures the intended model comparison under matched,
answer-containing context.

Answer-independent global FAISS top-5 retrieval, recall@1/5, and
retrieval-miss diagnostics remain a separate retrieval test. They are needed
to establish whether an ordinary retriever can deliver context comparable to
the oracle cache, but they are not required to interpret the matched reader
comparison reported here.

## Answer-independent FAISS condition

The corpus is the global union of the official Web and Wikipedia evidence files
listed in `unfiltered-web-dev.json`. The code follows the original simple
pipeline: collect filenames, load those files, split their text, and create one
FAISS index. Retrieval is not restricted to the files associated with the
current question: all selected documents are split into
300-word passages with 50-word overlap, embedded with
`sentence-transformers/all-MiniLM-L6-v2`, and searched through one FAISS index.
Gold answers and aliases are not used for retrieval.

This is deliberately an evaluation-conditioned, provided-evidence experiment.
It measures whether retrieval plus Qwen3-1.7B can use available knowledge; it
does not measure retrieval from an unseen, independently constructed corpus.

Files:

- `rag_for_triviaqa.py`: selects dev evidence, builds FAISS, and freezes top-5
  retrieval results;
- `prepare_oracle_context.py`: creates a reader upper-bound cache by selecting
  associated evidence chunks that contain a gold `NormalizedAlias`;
- `run_qwen_rag.py`: resumable no-thinking Qwen inference and EM/F1 scoring;
- `triviaqa.py`: TriviaQA answer normalization and metrics;
- `RUNBOOK.md`: exact preparation and execution commands.

Downloaded evidence, models, FAISS indexes, retrieval caches, and outputs belong
under `artifacts/` or `runs/` and must not be committed.
