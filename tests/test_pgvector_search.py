
from sentence_transformers import SentenceTransformer

from src.vector_database import search_legal_documents


MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

questions = [
    "What are the normal weekly working hours?",
    "What does the law say about salary information in employment contracts?",
    "What are the rules regarding overtime?"
]

for question in questions:

    print("\n" + "=" * 65)
    print("QUESTION:", question)
    print("=" * 65)

    question_embedding = model.encode(
        question,
        convert_to_numpy=True
    )

    results = search_legal_documents(
        question_embedding,
        top_k=5
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("Reference:", result["reference"])
        print("Similarity:", round(result["score"], 3))
        print("Text:", result["text"][:300])
