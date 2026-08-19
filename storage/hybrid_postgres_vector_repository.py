from typing import List

from document_parser import Chunk, EmbeddedChunk
from storage.postgres_vector_repository import PostgresVectorRepository
from storage.vector_repository import VectorRepository


class HybridPostgresVectorRepository(VectorRepository):
    """Dense (pgvector) + keyword (Postgres full-text) search fused with RRF."""

    def __init__(self, db_url: str, rrf_k: int = 60):
        self.dense = PostgresVectorRepository(db_url)
        self.conn = self.dense.conn
        self.rrf_k = rrf_k
        self._ensure_fts_index()

    def _ensure_fts_index(self) -> None:
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
            ON chunks
            USING gin (to_tsvector('english', content))
            """
        )
        self.conn.commit()

    def save_batch(self, chunks: List[EmbeddedChunk]) -> None:
        self.dense.save_batch(chunks)

    def get_last_chunk_index(self, filename: str) -> int | None:
        return self.dense.get_last_chunk_index(filename)

    def get_indexed_doc_ids(self) -> set[str]:
        return self.dense.get_indexed_doc_ids()

    def save_dlq(
        self,
        chunks: List[Chunk],
        error: str,
        attempts: int,
    ) -> None:
        self.dense.save_dlq(chunks, error=error, attempts=attempts)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        threshold: float | None = None,
        query: str | None = None,
    ) -> list[EmbeddedChunk]:
        # Fetch extra candidates from each channel, then fuse.
        fetch_limit = max(limit * 2, limit)
        dense_hits = self.dense.search(
            query_embedding,
            limit=fetch_limit,
            threshold=threshold,
        )
        if not query or not query.strip():
            return dense_hits[:limit]

        sparse_hits = self._keyword_search(query, limit=fetch_limit)
        return self._rrf_fuse(dense_hits, sparse_hits, limit=limit)

    def _keyword_search(self, query: str, limit: int) -> list[EmbeddedChunk]:
        sql = """
        SELECT content, chunk_index, metadata, embedding,
               ts_rank(
                   to_tsvector('english', content),
                   plainto_tsquery('english', %s)
               ) AS rank
        FROM chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (query, query, limit))
            return [
                EmbeddedChunk(
                    text=row[0],
                    index=row[1],
                    metadata=row[2],
                    embedding=PostgresVectorRepository._as_float_list(row[3]),
                    # Store rank as distance-like field for debugging; lower is better elsewhere.
                    distance=1.0 / (1.0 + float(row[4])),
                )
                for row in cursor.fetchall()
            ]

    def _rrf_fuse(
        self,
        dense_hits: list[EmbeddedChunk],
        sparse_hits: list[EmbeddedChunk],
        limit: int,
    ) -> list[EmbeddedChunk]:
        scores: dict[str, float] = {}
        chunks_by_key: dict[str, EmbeddedChunk] = {}

        def _key(chunk: EmbeddedChunk) -> str:
            doc_id = chunk.metadata.get("doc_id") or chunk.metadata.get("filename")
            return f"{doc_id}:{chunk.index}"

        for rank, chunk in enumerate(dense_hits, start=1):
            key = _key(chunk)
            chunks_by_key[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, chunk in enumerate(sparse_hits, start=1):
            key = _key(chunk)
            chunks_by_key[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered_keys = sorted(scores, key=scores.get, reverse=True)
        fused: list[EmbeddedChunk] = []
        for key in ordered_keys[:limit]:
            chunk = chunks_by_key[key]
            # Expose fused score in distance inverted for readability (lower still ~better).
            fused.append(
                EmbeddedChunk(
                    text=chunk.text,
                    index=chunk.index,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                    distance=1.0 / (1.0 + scores[key]),
                )
            )
        return fused
