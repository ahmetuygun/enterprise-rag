import time
from pathlib import Path

from chunker import Chunker
from document_parser import Chunk, DocumentCleaner, DocumentParser, EmbeddedChunk
from embedding.embedding_service import EmbeddingService
from storage.vector_repository import VectorRepository


class EmbeddingPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        chunker: Chunker,
        cleaner: DocumentCleaner | None = None,
        parser: DocumentParser | None = None,
        batch_size: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if base_delay <= 0:
            raise ValueError("base_delay must be > 0")
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.cleaner = cleaner
        self.parser = parser
        self.chunker = chunker
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay

    def run(self, file_path: str | Path) -> list[EmbeddedChunk]:
        path = Path(file_path)
        document = self.parser.parse(path)
        document = self.cleaner.clean(document)
        chunks = self.chunker.chunk(document)

        # Checkpoint: skip already-persisted chunks and resume from last index.
        filename = document.metadata.filename
        last_index = self.vector_repository.get_last_chunk_index(filename)
        if last_index is not None:
            chunks = [chunk for chunk in chunks if chunk.index > last_index]
            print(f"checkpoint resume after chunk_index={last_index}, remaining={len(chunks)}")

        if not chunks:
            print("nothing to embed")
            return []

        embedded_all: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embedded_batch = self._process_batch_with_retry(batch)
            if embedded_batch is None:
                continue
            embedded_all.extend(embedded_batch)
            print(
                f"saved batch chunk_index={batch[0].index}..{batch[-1].index} "
                f"({len(embedded_batch)} chunks)"
            )

        return embedded_all


    def run_json(self, corpus: dict[str, dict]) -> list[EmbeddedChunk]:
        if not hasattr(self.chunker, "chunk_corpus"):
            raise TypeError(
                "run_json requires a chunker with chunk_corpus(), "
                f"got {type(self.chunker).__name__}"
            )

        chunks = self.chunker.chunk_corpus(corpus)

        # Checkpoint: skip docs already stored (one chunk per doc_id).
        indexed = self.vector_repository.get_indexed_doc_ids()
        if indexed:
            before = len(chunks)
            chunks = [
                chunk
                for chunk in chunks
                if chunk.metadata.get("doc_id") not in indexed
            ]
            print(
                f"checkpoint skip indexed docs={before - len(chunks)}, "
                f"remaining={len(chunks)}"
            )

        if not chunks:
            print("nothing to embed")
            return []

        embedded_all: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embedded_batch = self._process_batch_with_retry(batch)
            if embedded_batch is None:
                continue
            embedded_all.extend(embedded_batch)
            doc_ids = [chunk.metadata.get("doc_id", "?") for chunk in batch]
            print(
                f"saved batch docs={doc_ids[0]}..{doc_ids[-1]} "
                f"({len(embedded_batch)} chunks)"
            )

        return embedded_all

    def _process_batch_with_retry(self, batch: list[Chunk]) -> list[EmbeddedChunk] | None:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                vectors = self.embedding_service.embed_batch([chunk.text for chunk in batch])
                embedded_batch = [
                    EmbeddedChunk(
                        text=chunk.text,
                        index=chunk.index,
                        metadata=chunk.metadata,
                        embedding=vector,
                    )
                    for chunk, vector in zip(batch, vectors)
                ]
                self.vector_repository.save_batch(embedded_batch)
                return embedded_batch
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Exponential backoff: base_delay * 2^(attempt-1)
                delay = self.base_delay * (2 ** (attempt - 1))
                print(
                    f"retry {attempt}/{self.max_retries} "
                    f"chunk_index={batch[0].index}..{batch[-1].index} "
                    f"after {delay:.1f}s: {exc}"
                )
                time.sleep(delay)

        error_msg = str(last_error) if last_error else "unknown error"
        self.vector_repository.save_dlq(batch, error=error_msg, attempts=self.max_retries)
        print(
            f"DLQ chunk_index={batch[0].index}..{batch[-1].index} "
            f"after {self.max_retries} attempts: {error_msg}"
        )
        return None


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from chunker import RecursiveDocumentChunker
    from document_parser import DocumentCleaner, DocumentParser
    from embedding.huggingface_embedding_service import HuggingFaceEmbeddingService
    from storage.postgres_vector_repository import PostgresVectorRepository

    load_dotenv()

    pipeline = EmbeddingPipeline(
        embedding_service=HuggingFaceEmbeddingService(api_key=os.environ["HF_TOKEN"]),
        vector_repository=PostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        cleaner=DocumentCleaner(),
        parser=DocumentParser(),
        chunker=RecursiveDocumentChunker(chunk_size=500),
        batch_size=5,
        max_retries=3,
        base_delay=1.0,
    )

    embedded = pipeline.run(Path("2608.06362v1.pdf"))
    print(f"embedded this run: {len(embedded)}")
