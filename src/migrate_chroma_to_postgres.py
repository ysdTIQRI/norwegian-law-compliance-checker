
from pathlib import Path

import chromadb
import psycopg

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

import os


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "legal_documents_cosine"

load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "norwegian_law_checker"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD")
}


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

def get_chroma_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ==========================================
# 3. MIGRATE DOCUMENTS
# ==========================================

def migrate_documents():

    print("\n========== CHROMADB TO POSTGRESQL MIGRATION ==========\n")

    collection = get_chroma_collection()

    print("ChromaDB collection:", collection.name)

    print("Documents in ChromaDB:", collection.count())

    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty.")

    data = collection.get(
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    records = []

    for document, metadata, embedding in zip(
        data["documents"],
        data["metadatas"],
        data["embeddings"]
    ):

        if embedding is None:
            raise ValueError(
                "A document is missing its embedding."
            )

        reference = metadata.get("reference")

        if not reference:
            raise ValueError(
                "A legal document is missing its reference."
            )

        if len(embedding) != 384:
            raise ValueError(
                f"Unexpected embedding dimension for {reference}: "
                f"{len(embedding)}"
            )

        records.append((
            document,
            "working_environment_act.txt",
            None,
            metadata.get("section"),
            reference,
            Vector(embedding.tolist())
        ))

    references = [record[4] for record in records]

    if len(references) != len(set(references)):
        raise ValueError(
            "Duplicate references found in ChromaDB."
        )

    print("\nDocuments prepared for migration:", len(records))

    # ==========================================
    # 4. CONNECT TO POSTGRESQL
    # ==========================================

    if not DB_CONFIG["password"]:
        raise ValueError(
            "DB_PASSWORD is missing from .env"
        )

    with psycopg.connect(**DB_CONFIG) as connection:

        register_vector(connection)

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

    print("\n========== MIGRATION COMPLETED ==========\n")

    verify_migration()


# ==========================================
# 5. VERIFY MIGRATION
# ==========================================

def verify_migration():

    with psycopg.connect(**DB_CONFIG) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(embedding),
                    COUNT(DISTINCT reference)
                FROM legal_documents;
                """
            )

            total, embedded, unique_references = cursor.fetchone()

    print("PostgreSQL legal documents:", total)
    print("Documents with embeddings:", embedded)
    print("Unique legal references:", unique_references)


# ==========================================
# 6. RUN MIGRATION
# ==========================================

if __name__ == "__main__":

    migrate_documents()
