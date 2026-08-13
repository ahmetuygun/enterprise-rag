"""Load BEIR SciFact corpus, queries, and qrels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.scifact.download import DATASET_DIR, download


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        if self.title and self.text:
            return f"{self.title}\n{self.text}"
        return self.title or self.text


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class SciFactSplit:
    corpus: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(data_dir: Path) -> dict[str, Document]:
    corpus: dict[str, Document] = {}
    for row in _load_jsonl(data_dir / "corpus.jsonl"):
        doc_id = str(row["_id"])
        corpus[doc_id] = Document(
            doc_id=doc_id,
            title=(row.get("title") or "").strip(),
            text=(row.get("text") or "").strip(),
        )
    return corpus


def load_queries(data_dir: Path) -> dict[str, Query]:
    queries: dict[str, Query] = {}
    for row in _load_jsonl(data_dir / "queries.jsonl"):
        query_id = str(row["_id"])
        queries[query_id] = Query(query_id=query_id, text=(row.get("text") or "").strip())
    return queries


def load_qrels(data_dir: Path, split: str = "test") -> dict[str, dict[str, int]]:
    path = data_dir / "qrels" / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"qrels not found: {path}")

    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if "query-id" not in header:
            handle.seek(0)

        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            query_id, doc_id, score = parts
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


def load_scifact(split: str = "test", data_dir: Path | None = None) -> SciFactSplit:
    root = data_dir or DATASET_DIR
    if not root.exists():
        root = download()

    corpus = load_corpus(root)
    queries = load_queries(root)
    qrels = load_qrels(root, split=split)

    # Keep only queries that have relevance labels.
    queries = {qid: query for qid, query in queries.items() if qid in qrels}

    print(
        f"loaded scifact/{split}: "
        f"corpus={len(corpus)} queries={len(queries)} "
        f"qrels={sum(len(v) for v in qrels.values())}"
    )
    return SciFactSplit(corpus=corpus, queries=queries, qrels=qrels)


if __name__ == "__main__":
    data = load_scifact(split="test")
    first_qid = next(iter(data.queries))
    print("sample query:", data.queries[first_qid].text)
    print("relevant docs:", data.qrels[first_qid])
