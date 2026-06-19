"""
db.py — Database initialization and connection management.
Creates the filings table with a pgvector column for semantic search.
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extensions import connection as PgConnection

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

POSTGRES_URL: str = os.getenv("POSTGRES_URL")


def get_connection() -> PgConnection:
    """Return a new psycopg2 connection to the Postgres database."""
    return psycopg2.connect(POSTGRES_URL)


def init_db() -> None:
    """
    Enable the pgvector extension and create the filings table if they don't exist.

    The filings table stores one row per text chunk, with:
    - Metadata: ticker, CIK, filing date, section name, chunk index
    - content: raw text of the chunk
    - embedding: 1024-dim vector from voyage-finance-2
    The hnsw index enables fast approximate nearest-neighbor search.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # pgvector extension must exist before we can use the vector type
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS filings (
                    id          SERIAL PRIMARY KEY,
                    ticker      TEXT NOT NULL,
                    cik         TEXT NOT NULL,
                    filed_date  DATE NOT NULL,
                    section     TEXT,
                    chunk_index INTEGER,
                    content     TEXT NOT NULL,
                    embedding   vector(1024)
                );
            """)

            # hnsw gives faster query time than ivfflat at the cost of more memory
            cur.execute("""
                CREATE INDEX IF NOT EXISTS filings_embedding_idx
                ON filings USING hnsw (embedding vector_cosine_ops);
            """)
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
