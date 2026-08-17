from document_parser import EmbeddedChunk


class ContextBuilder:
    """Turn reranked chunks into a single prompt context string."""

    def build(self, chunks: list[EmbeddedChunk]) -> str:
        if not chunks:
            return ""

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc_id = chunk.metadata.get("doc_id", "unknown")
            parts.append(f"[{i}] doc_id={doc_id}\n{chunk.text}")
        return "\n\n".join(parts)
