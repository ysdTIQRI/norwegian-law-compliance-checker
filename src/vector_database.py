
from pathlib import Path

import os
import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "all-MiniLM-L6-v2"

load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "norwegian_law_checker"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD")
}


# ==========================================
# 2. DATABASE CONNECTION
# ==========================================

def get_connection():

    if not DB_CONFIG["password"]:
        raise ValueError(
            "DB_PASSWORD is missing from your .env file."
        )

    connection = psycopg.connect(**DB_CONFIG)

    register_vector(connection)

    return connection


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

def load_embedding_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


# ==========================================
# 4. STORE LEGAL DOCUMENTS
# ==========================================

def store_legal_documents():

    # Import here to avoid circular imports
    try:
        from .legal_search import (
            load_legal_text,
            create_legal_chunks
        )
    except ImportError:
        from legal_search import (
            load_legal_text,
            create_legal_chunks
        )

    print("\n========== LEGAL DOCUMENT STORAGE ==========\n")

    print("Loading Norwegian legal information...")

    legal_text = load_legal_text()

    legal_chunks = create_legal_chunks(legal_text)

    print("Number of legal chunks:", len(legal_chunks))

    if not legal_chunks:
        raise ValueError("No legal chunks found.")

    # Check that every chunk has a unique reference
    references = [chunk["reference"] for chunk in legal_chunks]

    if len(references) != len(set(references)):
        raise ValueError(
            "Duplicate legal references found. "
            "Each legal section must have a unique reference."
        )

    # Load embedding model
    model = load_embedding_model()

    legal_texts = [
        chunk["text"]
        for chunk in legal_chunks
    ]

    print("\nGenerating legal embeddings...")

    embeddings = model.encode(
        legal_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("\nStoring legal documents in PostgreSQL...")

    records = []

    for chunk, embedding in zip(legal_chunks, embeddings):

        records.append((
            chunk["text"],
            "working_environment_act.txt",
            chunk.get("page_number"),
            chunk.get("section"),
            chunk["reference"],
            Vector(embedding.tolist())
        ))

    # Insert or update legal documents
    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.executemany(
                """
                INSERT INTO legal_documents (
                    content,
                    source,
                    page_number,
                    section,
                    reference,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)

                ON CONFLICT (reference)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    source = EXCLUDED.source,
                    page_number = EXCLUDED.page_number,
                    section = EXCLUDED.section,
                    embedding = EXCLUDED.embedding;
                """,
                records
            )

    print("\n========== STORAGE COMPLETED ==========\n")

    check_database()


# ==========================================
# 5. CHECK DATABASE INFORMATION
# ==========================================

def check_database():

    print("\n========== POSTGRESQL INFORMATION ==========\n")

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT COUNT(*) FROM legal_documents;"
            )

            total_documents = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM legal_documents
                WHERE embedding IS NOT NULL;
                """
            )

            embedded_documents = cursor.fetchone()[0]

    print("Database:", DB_CONFIG["dbname"])
    print("Table: legal_documents")
    print("Total legal documents:", total_documents)
    print("Documents with embeddings:", embedded_documents)


# ==========================================
# 6. RETRIEVE LEGAL DOCUMENTS
# ==========================================

def search_legal_documents(question_embedding, top_k=5):

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    content,
                    reference,
                    section,
                    embedding <=> %s AS distance

                FROM legal_documents

                WHERE embedding IS NOT NULL

                ORDER BY embedding <=> %s

                LIMIT %s;
                """,
                (
                    Vector(question_embedding.tolist()),
                    Vector(question_embedding.tolist()),
                    top_k
                )
            )

            rows = cursor.fetchall()

    results = []

    for row in rows:

        content, reference, section, distance = row

        results.append({
            "text": content,
            "reference": reference or section or "Unknown",
            "score": 1 - float(distance)
        })

    return results


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    store_legal_documents()

    check_database()
