
from sentence_transformers import SentenceTransformer

try:
    from .vector_database import search_legal_documents
except ImportError:
    from vector_database import search_legal_documents


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

_model = None


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

def get_embedding_model():

    global _model

    if _model is None:

        print("Loading embedding model...")

        _model = SentenceTransformer(MODEL_NAME)

    return _model


# ==========================================
# 3. SEARCH LEGAL INFORMATION
# ==========================================

def search_legal_information(question, top_k=DEFAULT_TOP_K):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    print("\n========== POSTGRESQL LEGAL SEARCH ==========\n")

    # Generate question embedding
    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        convert_to_numpy=True
    )

    # Search PostgreSQL using pgvector
    legal_results = search_legal_documents(
        question_embedding,
        top_k
    )

    if not legal_results:
        print("No legal documents found in PostgreSQL.")
        return []

    print("Retrieved legal documents:", len(legal_results))

    for rank, result in enumerate(legal_results, start=1):

        print(f"\nResult {rank}")

        print("Legal reference:", result["reference"])

        print(f"Similarity score: {result['score']:.3f}")

        print("Legal text:", result["text"])

        print("-" * 60)

    return legal_results


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    question = input(
        "Ask a question about Norwegian employment law: "
    ).strip()

    if question:
        search_legal_information(question)
    else:
        print("Please enter a question.")
