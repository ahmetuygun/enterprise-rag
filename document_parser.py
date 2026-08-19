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
    distance: float | None = None

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
    from storage.hybrid_postgres_vector_repository import HybridPostgresVectorRepository
    from embedding_pipeline import EmbeddingPipeline
    from retrieval_pipeline import RetrievalPipeline
    from evaluation.scifact.load import load_scifact
    from chunker import JsonDocumentChunker
    from rerank.bge_rerank_service import BGERerankService
    from context_builder import ContextBuilder
    from prompt_builder import PromptBuilder
    from llm.openai_llm_service import OpenAILLMService
    openai_api_key = os.environ["OPENAI_API_KEY"]
    hf_api_key = os.environ["HF_TOKEN"]

    corpus, queries, qrels = load_scifact()
    embedding_pipeline = EmbeddingPipeline(
        embedding_service=OpenAIEmbeddingService(api_key=openai_api_key),
        vector_repository=PostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        chunker=JsonDocumentChunker(),
        cleaner=None,
        parser=None,
        batch_size=5,
    )
    corpus = dict(list(corpus.items())[:5183])  # sadece ilk 100
    embedded = embedding_pipeline.run_json(corpus)
    print(f"embedded: {len(embedded)}")

    TOP_K = 20
    TOP_N = 5
    retrieval_pipeline = RetrievalPipeline(
        vector_repository=HybridPostgresVectorRepository(db_url=os.environ["DATABASE_URL"]),
        embedding_service=OpenAIEmbeddingService(api_key=openai_api_key),
        top_k=TOP_K,
        top_n=TOP_N,
        threshold=None,
        rerank_service=BGERerankService(),
    )

    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    llm = OpenAILLMService(api_key=openai_api_key)

    # Demo: second question -> context -> prompt -> answer
    sample_qid = list(qrels.keys())[0]
    sample_query = queries[sample_qid]
    results = retrieval_pipeline.retrieve(sample_query)
    context = context_builder.build(results)
    prompt = prompt_builder.build(sample_query, context)
    answer = llm.generate(prompt)

    print("=" * 80)
    print(f"QUERY ({sample_qid}): {sample_query}")
    print("=" * 80)
    print("\n--- context ---\n")
    print(context)
    print("\n--- prompt ---\n")
    print(prompt)
    print("\n--- answer ---\n")
    print(answer)

'''
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
        threshold=None,
    )
    queries = [
        "What is the main idea of the paper?",
        "What problem does the paper address?",
        "What is the main contribution of the paper?",
    ]
    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        results = retrieval_pipeline.retrieve(query=query)
        for i, chunk in enumerate(results, start=1):
            print(f"\n[{i}] distance={chunk.distance:.4f} chunk_index={chunk.index}")
            print(chunk.text)
'''

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
