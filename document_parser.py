import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter


load_dotenv()

@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict


@dataclass
class EmbeddedChunk:
    text: str
    index: int
    metadata: dict
    embedding: list[float]

@dataclass
class Page:
    number: int
    text: str


@dataclass
class DocumentMetadata:
    filename: str
    path: str
    page_count: int
    parser: str


@dataclass
class ParsedDocument:
    content: str
    metadata: DocumentMetadata
    pages: List[Page]


class DocumentParser:

    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, path: Path) -> ParsedDocument:

        result = self._converter.convert(path)

        pages = []

        for page_number, page in result.document.pages.items():

            # TODO: Docling page text extraction
            pages.append(
                Page(
                    number=page_number,
                    text=""
                )
            )

        metadata = DocumentMetadata(
            filename=path.name,
            path=str(path),
            page_count=len(result.document.pages),
            parser="docling"
        )

        return ParsedDocument(
            content=result.document.export_to_markdown(),
            metadata=metadata,
            pages=pages
        )

class DocumentCleaner:
    def clean(self, document: ParsedDocument) -> ParsedDocument:
        text = document.content
        text = text.replace("\r\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        text = "\n".join(lines)

        return ParsedDocument(
            content=text,
            metadata=document.metadata,
            pages=document.pages
        )
        

if __name__ == "__main__":
    from chunker import RecursiveDocumentChunker
    from embedding.bge_embedding_service import BGEEmbeddingService
    from embedding.openai_embedding_service import OpenAIEmbeddingService
    from embedding.huggingface_embedding_service import HuggingFaceEmbeddingService
    from storage.postgres_vector_repository import PostgresVectorRepository
    from embedding_pipeline import EmbeddingPipeline
    from retrieval_pipeline import RetrievalPipeline

    openai_api_key = os.environ["OPENAI_API_KEY"]
    hf_api_key = os.environ["HF_TOKEN"]
    
    pipeline = EmbeddingPipeline(
        embedding_service=HuggingFaceEmbeddingService(api_key=hf_api_key),
        vector_repository=PostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        cleaner=DocumentCleaner(),
        parser=DocumentParser(),
        chunker=RecursiveDocumentChunker(chunk_size=500),
        batch_size=5,
    )

    embedded = pipeline.run(Path("2608.06362v1.pdf"))
    print(f"embedded this run: {len(embedded)}")

    retrieval_pipeline = RetrievalPipeline(
        vector_repository=PostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        embedding_service=HuggingFaceEmbeddingService(api_key=hf_api_key),
        top_k=5,
        threshold=0.5
    )
    results = retrieval_pipeline.retrieve(query="What is the main idea of the paper?")
    print(f"results: {results}")

    # parser = DocumentParser()
    # document = parser.parse(Path("2608.06362v1.pdf"))
    # cleaner = DocumentCleaner()
    # document = cleaner.clean(document)

    # chunker = RecursiveDocumentChunker(chunk_size=500)
    # chunks = chunker.chunk(document)

    # sample = chunks[:5]
    # texts = [chunk.text for chunk in sample]

    # # embedding_service = BGEEmbeddingService(model="BAAI/bge-small-en-v1.5")
    # # vectors = embedding_service.embed_batch(texts)
    # # embedding_service = OpenAIEmbeddingService(api_key=api_key)
    # embedding_service = HuggingFaceEmbeddingService(api_key=hf_api_key)
    # vectors = embedding_service.embed_batch(texts)

    # embedded_chunks = [
    #     EmbeddedChunk(
    #         text=chunk.text,
    #         index=chunk.index,
    #         metadata=chunk.metadata,
    #         embedding=vector,
    #     )
    #     for chunk, vector in zip(sample, vectors)
    # ]

    # repository = PostgresVectorRepository(db_url=os.environ["DATABASE_URL"])
    # repository.save_batch(embedded_chunks)

    # print(f"chunks: {len(chunks)}")
    # print(f"embedded: {len(embedded_chunks)}")
    # print(f"dims: {len(embedded_chunks[0].embedding)}")
    # print(embedded_chunks[0].text[:100])
    # print(embedded_chunks[0].embedding[:8])
