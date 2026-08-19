class PromptBuilder:
    """Build a simple RAG prompt from query + context."""

    def build(self, query: str, context: str) -> str:
        return (
            "Answer the question using only the context below.\n"
            "If the context is not enough, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
