from embedding.embedding_service import EmbeddingService
from openai import OpenAI
from typing import List


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_batch(self, texts: List[str]) -> List[float]:
        return self.api_key.embed(texts)