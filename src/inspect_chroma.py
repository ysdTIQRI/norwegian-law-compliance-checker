from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "legal_documents_cosine"

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Total legal chunks:", collection.count())

data = collection.get(
    include=["documents", "metadatas"]
)

for index, (document, metadata) in enumerate(
    zip(data["documents"], data["metadatas"]),
    start=1
):
    print("\n" + "=" * 70)
    print("CHUNK:", index)
    print("REFERENCE:", metadata.get("reference"))
    print("SECTION:", metadata.get("section"))
    print("CONTENT:")
    print(document)