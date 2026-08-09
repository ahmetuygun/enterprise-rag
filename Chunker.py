from typing import List, Protocol
from document_parser import Chunk, ParsedDocument


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        ...


class FixedSizeDocumentChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.content
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        index = 0
        metadata = {
            "filename": document.metadata.filename,
            "strategy": "fixed",
        }

        while start < len(text):
            end = start + self.chunk_size
            part = text[start:end]
            chunks.append(
                Chunk(
                    index=index,
                    metadata=metadata,
                    text=part,
                )
            )
            start = end - self.overlap
            index += 1

        return chunks


class RecursiveDocumentChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        # overlap kept for a consistent API; not used in this simple version
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.content
        if not text:
            return []

        pieces = self._split(text, self.separators)
        pieces = self._merge(pieces)
        metadata = {
            "filename": document.metadata.filename,
            "strategy": "recursive",
        }

        return [
            Chunk(text=piece, index=index, metadata=metadata)
            for index, piece in enumerate(pieces)
        ]

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not text.strip():
            return []

        # Last resort: hard cut by characters
        if not separators or separators[0] == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = separators[0]
        rest = separators[1:]

        if len(text) <= self.chunk_size:
            return [text]

        parts = text.split(separator)
        result: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._split(part, rest))

        return result

    def _merge(self, pieces: List[str]) -> List[str]:
        """Combine tiny splits until each chunk is close to chunk_size."""
        if not pieces:
            return []

        merged: List[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            candidate = f"{current} {piece}"
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current)
                current = piece

        merged.append(current)
        return merged


class MarkdownDocumentChunker:
    """Split on markdown headings; oversized sections fall back to recursive split."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._recursive = RecursiveDocumentChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.content
        if not text:
            return []

        sections = self._split_by_headings(text)
        metadata = {
            "filename": document.metadata.filename,
            "strategy": "markdown",
        }

        chunks: List[Chunk] = []
        index = 0

        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(
                    Chunk(text=section, index=index, metadata=metadata)
                )
                index += 1
            else:
                pieces = self._recursive._split(section, self._recursive.separators)
                pieces = self._recursive._merge(pieces)
                for piece in pieces:
                    chunks.append(
                        Chunk(text=piece, index=index, metadata=metadata)
                    )
                    index += 1

        return chunks

    def _split_by_headings(self, text: str) -> List[str]:
        lines = text.splitlines()
        sections: List[str] = []
        current: List[str] = []

        for line in lines:
            if line.startswith("#") and current:
                section = "\n".join(current).strip()
                if section:
                    sections.append(section)
                current = [line]
            else:
                current.append(line)

        if current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)

        return sections
