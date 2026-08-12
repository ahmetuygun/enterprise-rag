from pathlib import Path

from chunker import Chunker
from document_parser import DocumentCleaner, DocumentParser, EmbeddedChunk
from embedding.embedding_service import EmbeddingService
from storage.vector_repository import VectorRepository


class EmbeddingPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        cleaner: DocumentCleaner,
        parser: DocumentParser,
        chunker: Chunker,
        batch_size: int = 5,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.cleaner = cleaner
        self.parser = parser
        self.chunker = chunker
        self.batch_size = batch_size

    def run(self, file_path: str | Path) -> list[EmbeddedChunk]:
        path = Path(file_path)
        document = self.parser.parse(path)
        document = self.cleaner.clean(document)
        chunks = self.chunker.chunk(document)

        filename = document.metadata.filename
        last_index = self.vector_repository.get_last_chunk_index(filename)
        if last_index is not None:
            chunks = [chunk for chunk in chunks if chunk.index > last_index]
            print(f"resume after chunk_index={last_index}, remaining={len(chunks)}")

        if not chunks:
            print("nothing to embed")
            return []

        embedded_all: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
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
            embedded_all.extend(embedded_batch)
            print(
                f"saved batch chunk_index={batch[0].index}..{batch[-1].index} "
                f"({len(embedded_batch)} chunks)"
            )

        return embedded_all


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
    )

    embedded = pipeline.run(Path("2608.06362v1.pdf"))
    print(f"embedded this run: {len(embedded)}")
