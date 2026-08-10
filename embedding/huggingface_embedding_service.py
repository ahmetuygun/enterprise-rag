from embedding.embedding_service import EmbeddingService
from huggingface_hub import InferenceClient
from typing import List

class HuggingFaceEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "BAAI/bge-large-en-v1.5"):
        self.api_key = api_key
        self.model = model
        self.client = InferenceClient(api_key=api_key)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.client.feature_extraction(texts, model=self.model, normalize=True)
        return vectors.tolist()
