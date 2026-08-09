from typing import List

from sentence_transformers import SentenceTransformer

from embedding.embedding_service import EmbeddingService


class BGEEmbeddingService(EmbeddingService):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
