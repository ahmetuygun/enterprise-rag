from typing import List, Protocol

from document_parser import EmbeddedChunk


class VectorRepository(Protocol):
    def save_batch(self, chunks: List[EmbeddedChunk]) -> None:
        ...

    def search(self, query_embedding: list[float], limit: int = 5) -> list[EmbeddedChunk]:
        ...

    def get_last_chunk_index(self, filename: str) -> int | None:
        ...
