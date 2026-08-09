from embedding.embedding_service import EmbeddingService
from typing import List

class HuggingFaceEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "bge-large-en-v1.5"):
        self.api_key = api_key

    def embed_batch(self, texts: List[str]) -> List[float]:
        return self.api_key.embed(texts)
        