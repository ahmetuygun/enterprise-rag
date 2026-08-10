from typing import List, Protocol

class EmbeddingService(Protocol):
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...


