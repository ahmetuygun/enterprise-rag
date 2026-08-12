from typing import List, Protocol

from document_parser import Chunk, EmbeddedChunk


class VectorRepository(Protocol):
    def save_batch(self, chunks: List[EmbeddedChunk]) -> None:
        ...

    def get_last_chunk_index(self, filename: str) -> int | None:
        ...

    def save_dlq(
        self,
        chunks: List[Chunk],
        error: str,
        attempts: int,
    ) -> None:
        ...

    def search(self, query_embedding: list[float], limit: int = 5) -> list[EmbeddedChunk]:
        ...
