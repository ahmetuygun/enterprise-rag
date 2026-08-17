import os

from dotenv import load_dotenv

from embedding.openai_embedding_service import OpenAIEmbeddingService
from evaluation.scifact.load import load_scifact
from retrieval_pipeline import RetrievalPipeline
from storage.postgres_vector_repository import PostgresVectorRepository

load_dotenv()

TOP_K = 5
MAX_QUERIES = 20  # keep small while testing


def main() -> None:
    _, queries, qrels = load_scifact(split="test")

    retrieval = RetrievalPipeline(
        vector_repository=PostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        embedding_service=OpenAIEmbeddingService(api_key=os.environ["OPENAI_API_KEY"]),
        top_k=TOP_K,
        threshold=None,
    )

    hits = 0
    total = 0

    for qid in list(qrels.keys())[:MAX_QUERIES]:
        query = queries[qid]
        gold = set(qrels[qid].keys())

        results = retrieval.retrieve(query)
        retrieved = {chunk.metadata.get("doc_id") for chunk in results}

        found = gold & retrieved
        ok = len(found) > 0
        hits += int(ok)
        total += 1

        print(f"qid={qid} {'HIT' if ok else 'MISS'} gold={gold} got={retrieved}")

    print(f"\nHIT@{TOP_K}: {hits}/{total} = {hits / total if total else 0:.3f}")


if __name__ == "__main__":
    main()
