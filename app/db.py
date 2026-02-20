import psycopg2
from app.config import DB_CONFIG


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR(384),
                source TEXT,
                chunk_index INT
            );
            """)
        conn.commit()
