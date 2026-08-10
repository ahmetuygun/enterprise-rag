from document_parser import DocumentParser, DocumentCleaner
from chunker import Chunker
from embedding.embedding_service import EmbeddingService
from storage.vector_repository import VectorRepository
from document_parser import EmbeddedChunk


class EmbeddingPipeline:
    def __init__(self, embedding_service: EmbeddingService, 
    vector_repository: VectorRepository,
    cleaner: DocumentCleaner,
    parser: DocumentParser,
    chunker: Chunker):
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.cleaner = cleaner
        self.parser = parser
        self.chunker = chunker

    def run(self, file_path: str) -> list[EmbeddedChunk]:
        document = self.parser.parse(file_path)
        document = self.cleaner.clean(document)
        chunks = self.chunker.chunk(document)
        vectors = self.embedding_service.embed_batch([chunk.text for chunk in chunks])
       
        embedded_chunks = [
         EmbeddedChunk(text=c.text, index=c.index, metadata=c.metadata, embedding=v)
         for c, v in zip(chunks, vectors)
        ]
    
        self.vector_repository.save_batch(embedded_chunks)
        return embedded_chunks