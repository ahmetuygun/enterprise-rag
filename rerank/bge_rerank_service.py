from sentence_transformers import CrossEncoder

from document_parser import EmbeddedChunk
from rerank.rerank_service import RerankService


class BGERerankService(RerankService):
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        documents: list[EmbeddedChunk],
        top_n: int = 5,
    ) -> list[EmbeddedChunk]:
        if not documents:
            return []

        pairs = [(query, chunk.text) for chunk in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [chunk for chunk, _ in ranked[:top_n]]
