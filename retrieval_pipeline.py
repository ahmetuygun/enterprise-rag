from document_parser import EmbeddedChunk
from embedding.embedding_service import EmbeddingService
from rerank.rerank_service import RerankService
from storage.vector_repository import VectorRepository


class RetrievalPipeline:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        top_k: int = 20,
        top_n: int = 5,
        threshold: float | None = None,
        rerank_service: RerankService | None = None,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.top_n = top_n
        self.threshold = threshold
        self.rerank_service = rerank_service

    def retrieve(self, query: str) -> list[EmbeddedChunk]:
        query_embedding = self.embedding_service.embed_batch([query])[0]
        candidates = self.vector_repository.search(
            query_embedding,
            limit=self.top_k,
            threshold=self.threshold,
            query=query,
        )
        if self.rerank_service is None:
            return candidates[: self.top_n]
        return self.rerank_service.rerank(query, candidates, top_n=self.top_n)
