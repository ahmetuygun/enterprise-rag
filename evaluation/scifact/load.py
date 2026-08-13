from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent / "data" / "scifact"


def load_corpus(data_dir: Path) -> dict:
    corpus = {}
    with (data_dir / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}
    return corpus


def load_queries(data_dir: Path) -> dict:
    queries = {}
    with (data_dir / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        next(f)  # skip header
        for line in f:
            qid, doc_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def load_scifact(data_dir: Path = DATA_DIR, split: str = "test"):
    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_qrels(data_dir / "qrels" / f"{split}.tsv")
    return corpus, queries, qrels


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    print(len(corpus), len(queries), len(qrels))
    print(next(iter(qrels.items())))
