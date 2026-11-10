#!/usr/bin/env python3
"""Create a FAISS index from TriviaQA evidence and retrieve context for Qwen."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def select_files(evidence_path, json_path):
    """Select the evidence files named in unfiltered-web-dev.json."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)

    files = []
    for example in data["Data"]:
        for item in example.get("EntityPages", []):
            filename = item.get("Filename")
            if filename:
                files.append(os.path.join(evidence_path, "wikipedia", filename))
        for item in example.get("SearchResults", []):
            filename = item.get("Filename")
            if filename:
                files.append(os.path.join(evidence_path, "web", filename))

    # The same evidence file can be referenced by several questions.
    return list(dict.fromkeys(files))


def load_langchain_documents(files):
    documents = []
    for filename in files:
        with open(filename, encoding="utf-8", errors="replace") as file:
            documents.append(
                Document(page_content=file.read(), metadata={"source": filename})
            )
    return documents


def create_vector_db(
    files,
    output_path,
    embedding_model=EMBEDDING_MODEL,
    embedding_revision=None,
):
    documents = load_langchain_documents(files)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=lambda text: len(text.split()),
    )
    chunks = splitter.split_documents(documents)

    model_kwargs = {}
    if embedding_revision:
        model_kwargs["revision"] = embedding_revision
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
    )
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(output_path)
    return vector_db


def load_vector_db(index_path, embedding_model=EMBEDDING_MODEL, embedding_revision=None):
    model_kwargs = {}
    if embedding_revision:
        model_kwargs["revision"] = embedding_revision
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
    )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_docs(text, vector_db, top_k=5):
    # The original version accidentally searched for a fixed sentence here.
    return vector_db.similarity_search(text, k=top_k)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_retrieval_results(dataset_path, vector_db, index_path, output_path, top_k=5):
    """Save retrieved passages once so the same context is used for every Qwen run."""
    with open(dataset_path, encoding="utf-8") as file:
        data = json.load(file)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for example in data["Data"]:
            question = example["Question"]
            passages = retrieve_docs(question, vector_db, top_k)
            record = {
                "question_id": example["QuestionId"],
                "question": question,
                "passages": [
                    {
                        "passage_id": hashlib.sha256(
                            (document.metadata["source"] + "\0" + document.page_content).encode()
                        ).hexdigest(),
                        "source": document.metadata["source"],
                        "title": Path(document.metadata["source"]).stem,
                        "text": document.page_content,
                    }
                    for document in passages
                ],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "mode": "faiss",
        "top_k": top_k,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "index_sha256": sha256(Path(index_path) / "index.faiss"),
        "gold_used_for_retrieval": False,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--evidence-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-existing-index", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_existing_index:
        vector_db = load_vector_db(
            args.index_path,
            args.embedding_model,
            args.embedding_revision,
        )
    else:
        files = select_files(args.evidence_path, args.dataset)
        print(f"Selected {len(files)} evidence files")
        vector_db = create_vector_db(
            files,
            args.index_path,
            args.embedding_model,
            args.embedding_revision,
        )
    save_retrieval_results(args.dataset, vector_db, args.index_path, args.output, args.top_k)


if __name__ == "__main__":
    main()
