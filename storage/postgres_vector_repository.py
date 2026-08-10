from typing import List
from document_parser import EmbeddedChunk
from storage.vector_repository import VectorRepository
from pgvector.psycopg import register_vector
import psycopg
from psycopg.types.json import Jsonb

class PostgresVectorRepository(VectorRepository):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = psycopg.connect(self.db_url)
        register_vector(self.conn)

    def save_batch(self, chunks: List[EmbeddedChunk]) -> None:
        sql = """
        INSERT INTO chunks (content, chunk_index, metadata, embedding)
        VALUES (%s, %s, %s::jsonb, %s)
        """
        with self.conn.cursor() as cursor:
            for chunk in chunks:
                cursor.execute(sql, (chunk.text, chunk.index, Jsonb(chunk.metadata), chunk.embedding ))
            self.conn.commit()

    def search(self, query_embedding: list[float], limit: int = 5) -> list[EmbeddedChunk]:
        sql = """
        SELECT content, chunk_index, metadata, embedding
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (query_embedding, limit))
            return [EmbeddedChunk(text=row[0], index=row[1], metadata=row[2], embedding=row[3]) for row in cursor.fetchall()]