from typing import Protocol

from document_parser import EmbeddedChunk


class RerankService(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[EmbeddedChunk],
        top_n: int = 5,
    ) -> list[EmbeddedChunk]:
        ...
