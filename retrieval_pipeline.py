from embedding.embedding_service import EmbeddingService
from document_parser import EmbeddedChunk
from storage.vector_repository import VectorRepository


class RetrievalPipeline:
    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        threshold: float | None = None,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> list[EmbeddedChunk]:
        query_embedding = self.embedding_service.embed_batch([query])[0]
        return self.vector_repository.search(
            query_embedding,
            limit=self.top_k,
            threshold=self.threshold,
        )
