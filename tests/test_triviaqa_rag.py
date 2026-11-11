from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.documents import Document


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "issue-11-triviaqa-rag"
sys.path.insert(0, str(EXPERIMENT_DIR))

from rag_for_triviaqa import retrieve_docs, select_files  # noqa: E402
from prepare_oracle_context import create_oracle_cache  # noqa: E402
from run_qwen_rag import context_has_answer, render_task, retrieval_answer_rank  # noqa: E402


def write_dataset(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps({"Version": 1.0, "Data": data}), encoding="utf-8")


def test_selects_web_and_wikipedia_files_from_dev(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "web").mkdir(parents=True)
    (evidence / "wikipedia").mkdir()
    (evidence / "web" / "france.txt").write_text("Paris is in France.", encoding="utf-8")
    (evidence / "wikipedia" / "ocean.txt").write_text("Pacific Ocean.", encoding="utf-8")
    dataset = tmp_path / "dev.json"
    write_dataset(
        dataset,
        [
            {
                "QuestionId": "dev-1",
                "Question": "What is the capital of France?",
                "SearchResults": [{"Filename": "france.txt", "Title": "France"}],
                "EntityPages": [{"Filename": "ocean.txt", "Title": "Pacific"}],
            }
        ],
    )

    selected = select_files(str(evidence), str(dataset))

    assert selected == [
        str(evidence / "wikipedia" / "ocean.txt"),
        str(evidence / "web" / "france.txt"),
    ]


def test_retrieval_uses_supplied_question_and_builds_prompt() -> None:
    class FakeVectorDB:
        def __init__(self) -> None:
            self.query = None

        def similarity_search(self, query: str, k: int):
            self.query = query
            assert k == 1
            return [
                Document(
                    page_content="Paris is the capital of France.",
                    metadata={"source": "web/france.txt"},
                )
            ]

    vector_db = FakeVectorDB()
    question = "What is the capital of France?"
    documents = retrieve_docs(question, vector_db, top_k=1)
    passages = [
        {
            "text": document.page_content,
            "source": document.metadata["source"],
            "passage_id": "test",
            "title": "France",
        }
        for document in documents
    ]

    assert vector_db.query == question
    assert context_has_answer(passages, ("paris",))
    assert retrieval_answer_rank(passages, ("paris",)) == 1
    prompt = render_task(question, passages)
    assert "Paris is the capital of France." in prompt
    assert prompt.endswith(f"Question: {question}\nAnswer:")


def test_oracle_cache_selects_answer_chunk_and_marks_fallback(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "web").mkdir(parents=True)
    (evidence / "wikipedia").mkdir()
    (evidence / "web" / "france.txt").write_text(
        "France is in Europe. Paris is the capital of France.", encoding="utf-8"
    )
    (evidence / "web" / "ocean.txt").write_text(
        "This document contains no accepted answer.", encoding="utf-8"
    )
    dataset = tmp_path / "dev.json"
    write_dataset(
        dataset,
        [
            {
                "QuestionId": "found",
                "Question": "What is the capital of France?",
                "Answer": {"NormalizedAliases": ["paris"], "HumanAnswers": ["Paris"]},
                "SearchResults": [{"Filename": "france.txt", "Title": "France"}],
                "EntityPages": [],
            },
            {
                "QuestionId": "fallback",
                "Question": "Which answer is absent?",
                "Answer": {"NormalizedAliases": ["missing"], "HumanAnswers": ["Missing"]},
                "SearchResults": [{"Filename": "ocean.txt", "Title": "Ocean"}],
                "EntityPages": [],
            },
        ],
    )
    output = tmp_path / "oracle.jsonl"

    metadata = create_oracle_cache(dataset, evidence, output, top_k=5)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert records[0]["oracle_answer_found"] is True
    assert "Paris" in records[0]["passages"][0]["text"]
    assert records[1]["oracle_answer_found"] is False
    assert metadata["answer_found_examples"] == 1
    assert metadata["fallback_examples"] == 1
