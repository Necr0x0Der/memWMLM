# Issue #11: provided-evidence TriviaQA RAG

GitHub issue: https://github.com/Necr0x0Der/memWMLM/issues/11

This experiment asks whether Qwen3-1.7B can use provided factual knowledge to
close part of its closed-book TriviaQA gap to Qwen3-4B. It is an exploratory
RAG/context baseline, not a test of the native memWMLM sidecar.

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
- `run_qwen_rag.py`: resumable no-thinking Qwen inference and EM/F1 scoring;
- `triviaqa.py`: TriviaQA answer normalization and metrics;
- `RUNBOOK.md`: exact preparation and execution commands.

Downloaded evidence, models, FAISS indexes, retrieval caches, and outputs belong
under `artifacts/` or `runs/` and must not be committed.
