"""Dense-retrieval evaluation on SciFact (nDCG / Recall / MRR)."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

import numpy as np
from dotenv import load_dotenv

from embedding.embedding_service import EmbeddingService
from evaluation.scifact.load import SciFactSplit, load_scifact


def _embed_texts(
    embedding_service: EmbeddingService,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        vectors.extend(embedding_service.embed_batch(batch))
        done = min(start + batch_size, len(texts))
        print(f"embedded {done}/{len(texts)}")
    return np.asarray(vectors, dtype=np.float32)


def _dcg(relevances: Sequence[float], k: int) -> float:
    score = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        score += float(rel) / math.log2(rank + 1)
    return score


def ndcg_at_k(qrels: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    gains = [float(qrels.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    dcg = _dcg(gains, k)
    ideal = sorted((float(v) for v in qrels.values()), reverse=True)
    idcg = _dcg(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def recall_at_k(qrels: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(qrels: dict[str, int], ranked_ids: Sequence[str], k: int) -> float:
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_rankings(
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    metrics = {f"{name}@{k}": 0.0 for name in ("ndcg", "recall", "mrr") for k in ks}
    query_ids = [qid for qid in rankings if qid in qrels]
    if not query_ids:
        return metrics

    for query_id in query_ids:
        ranked = rankings[query_id]
        labels = qrels[query_id]
        for k in ks:
            metrics[f"ndcg@{k}"] += ndcg_at_k(labels, ranked, k)
            metrics[f"recall@{k}"] += recall_at_k(labels, ranked, k)
            metrics[f"mrr@{k}"] += mrr_at_k(labels, ranked, k)

    n = len(query_ids)
    return {name: value / n for name, value in metrics.items()}


def retrieve_all(
    data: SciFactSplit,
    embedding_service: EmbeddingService,
    top_k: int,
    batch_size: int,
) -> dict[str, list[str]]:
    doc_ids = list(data.corpus.keys())
    doc_texts = [data.corpus[doc_id].content for doc_id in doc_ids]
    print(f"embedding corpus ({len(doc_texts)} docs), batch_size={batch_size}")
    doc_matrix = _embed_texts(embedding_service, doc_texts, batch_size=batch_size)

    # Normalized embeddings => cosine similarity via dot product.
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    doc_matrix = doc_matrix / np.clip(doc_norms, 1e-12, None)

    query_ids = list(data.queries.keys())
    query_texts = [data.queries[qid].text for qid in query_ids]
    print(f"embedding queries ({len(query_texts)}), batch_size={batch_size}")
    query_matrix = _embed_texts(embedding_service, query_texts, batch_size=batch_size)
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    query_matrix = query_matrix / np.clip(query_norms, 1e-12, None)

    print("scoring...")
    scores = query_matrix @ doc_matrix.T
    rankings: dict[str, list[str]] = {}
    for row_idx, query_id in enumerate(query_ids):
        top_idx = np.argpartition(-scores[row_idx], top_k - 1)[:top_k]
        ordered = top_idx[np.argsort(-scores[row_idx, top_idx])]
        rankings[query_id] = [doc_ids[i] for i in ordered]
    return rankings


def run_evaluation(
    embedding_service: EmbeddingService,
    split: str = "test",
    top_k: int = 10,
    batch_size: int = 32,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    data = load_scifact(split=split)
    rankings = retrieve_all(
        data,
        embedding_service=embedding_service,
        top_k=top_k,
        batch_size=batch_size,
    )
    metrics = evaluate_rankings(data.qrels, rankings, ks=ks)

    print("\nSciFact metrics")
    for key in sorted(metrics, key=lambda name: (name.split("@")[0], int(name.split("@")[1]))):
        print(f"{key:12s} {metrics[key]:.4f}")
    return metrics


def _build_embedding_service() -> EmbeddingService:
    from embedding.huggingface_embedding_service import HuggingFaceEmbeddingService

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for SciFact evaluation")
    return HuggingFaceEmbeddingService(api_key=token)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate dense retrieval on SciFact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_evaluation(
        embedding_service=_build_embedding_service(),
        split=args.split,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
