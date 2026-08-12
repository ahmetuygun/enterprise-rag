from typing import List

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from document_parser import Chunk, EmbeddedChunk
from storage.vector_repository import VectorRepository


class PostgresVectorRepository(VectorRepository):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = psycopg.connect(self.db_url)
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # Lets resume skips / re-runs avoid duplicate (filename, chunk_index) rows.
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS chunks_filename_chunk_index_uidx
            ON chunks ((metadata->>'filename'), chunk_index)
            """
        )
        # Cosine distance (<=>) queries need vector_cosine_ops.
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
            ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_letter_chunks (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL,
                error TEXT NOT NULL,
                attempts INT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        self.conn.commit()

    def save_batch(self, chunks: List[EmbeddedChunk]) -> None:
        if not chunks:
            return

        sql = """
        INSERT INTO chunks (content, chunk_index, metadata, embedding)
        VALUES (%s, %s, %s::jsonb, %s)
        ON CONFLICT ((metadata->>'filename'), chunk_index) DO NOTHING
        """
        with self.conn.cursor() as cursor:
            for chunk in chunks:
                cursor.execute(
                    sql,
                    (
                        chunk.text,
                        chunk.index,
                        Jsonb(chunk.metadata),
                        chunk.embedding,
                    ),
                )
            self.conn.commit()

    def get_last_chunk_index(self, filename: str) -> int | None:
        row = self.conn.execute(
            """
            SELECT MAX(chunk_index)
            FROM chunks
            WHERE metadata->>'filename' = %s
            """,
            (filename,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def save_dlq(
        self,
        chunks: List[Chunk],
        error: str,
        attempts: int,
    ) -> None:
        if not chunks:
            return

        sql = """
        INSERT INTO dead_letter_chunks
            (filename, chunk_index, content, metadata, error, attempts)
        VALUES (%s, %s, %s, %s::jsonb, %s, %s)
        """
        with self.conn.cursor() as cursor:
            for chunk in chunks:
                filename = chunk.metadata.get("filename", "unknown")
                cursor.execute(
                    sql,
                    (
                        filename,
                        chunk.index,
                        chunk.text,
                        Jsonb(chunk.metadata),
                        error,
                        attempts,
                    ),
                )
            self.conn.commit()

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[EmbeddedChunk]:
        # threshold = max cosine distance (<=>). Lower is more similar.
        # None = no distance filter.
        if threshold is not None and threshold < 0:
            raise ValueError("threshold must be >= 0")

        if threshold is None:
            sql = """
            SELECT content, chunk_index, metadata, embedding
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """
            params = (query_embedding, limit)
        else:
            sql = """
            SELECT content, chunk_index, metadata, embedding
            FROM chunks
            WHERE embedding <=> %s <= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
            params = (query_embedding, threshold, query_embedding, limit)

        with self.conn.cursor() as cursor:
            cursor.execute(sql, params)
            return [
                EmbeddedChunk(
                    text=row[0],
                    index=row[1],
                    metadata=row[2],
                    embedding=list(row[3]),
                )
                for row in cursor.fetchall()
            ]
