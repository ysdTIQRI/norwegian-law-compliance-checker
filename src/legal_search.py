
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from .vector_database import search_legal_documents
except ImportError:
    from vector_database import search_legal_documents


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

LEGAL_DATA_FILE = (
    BASE_DIR / "legal_data" / "working_environment_act.txt"
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

_model = None


# ==========================================
# 2. LOAD NORWEGIAN LEGAL INFORMATION
# ==========================================

def load_legal_text():

    print("\nLoading Norwegian legal information...")

    if not LEGAL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Legal data file not found: {LEGAL_DATA_FILE}"
        )

    legal_text = LEGAL_DATA_FILE.read_text(
        encoding="utf-8"
    )

    if not legal_text.strip():
        raise ValueError(
            "The Norwegian legal information file is empty."
        )

    print("Legal information loaded successfully.")

    return legal_text


# ==========================================
# 3. CREATE LEGAL DOCUMENT CHUNKS
# ==========================================

def create_legal_chunks(legal_text):

    if not legal_text or not legal_text.strip():
        raise ValueError("Legal text cannot be empty.")

    chunks = []

    # Identify legal sections
    section_pattern = re.compile(
        r"(?m)^Section\s+(14-\d+)\s+-.*$"
    )

    section_matches = list(
        section_pattern.finditer(legal_text)
    )

    if not section_matches:
        raise ValueError(
            "No legal sections found in the legal text."
        )

    # Process each legal section
    for index, section_match in enumerate(section_matches):

        section_number = section_match.group(1)

        section_start = section_match.end()

        section_end = (
            section_matches[index + 1].start()
            if index + 1 < len(section_matches)
            else len(legal_text)
        )

        section_text = legal_text[
            section_start:section_end
        ].strip()

        # Identify numbered subsections: (1), (2), (3), etc.
        subsection_pattern = re.compile(
            r"(?m)^\((\d+)\)\s*"
        )

        subsection_matches = list(
            subsection_pattern.finditer(section_text)
        )

        for sub_index, sub_match in enumerate(
            subsection_matches
        ):

            subsection_number = sub_match.group(1)

            subsection_start = sub_match.end()

            subsection_end = (
                subsection_matches[sub_index + 1].start()
                if sub_index + 1 < len(subsection_matches)
                else len(section_text)
            )

            subsection_text = section_text[
                subsection_start:subsection_end
            ].strip()

            base_reference = (
                f"{section_number}({subsection_number})"
            )

            # ------------------------------------------
            # Special handling for Section 14-6 (1)
            # ------------------------------------------

            # This subsection contains lettered clauses (a) to (q).
            if (
                section_number == "14-6"
                and subsection_number == "1"
            ):

                letter_pattern = re.compile(
                    r"(?m)^\(([a-q])\)\s*"
                )

                letter_matches = list(
                    letter_pattern.finditer(subsection_text)
                )

                if not letter_matches:
                    raise ValueError(
                        "Could not find lettered clauses "
                        "in Section 14-6 (1)."
                    )

                # Extract introductory text
                introduction = subsection_text[
                    :letter_matches[0].start()
                ].strip()

                # Store the introductory requirement
                if introduction:

                    chunks.append({
                        "text": introduction,
                        "reference": base_reference,
                        "section": section_number,
                        "page_number": None
                    })

                # Extract individual clauses (a) to (q)
                for letter_index, letter_match in enumerate(
                    letter_matches
                ):

                    letter = letter_match.group(1)

                    clause_start = letter_match.end()

                    clause_end = (
                        letter_matches[letter_index + 1].start()
                        if letter_index + 1 < len(letter_matches)
                        else len(subsection_text)
                    )

                    clause_text = subsection_text[
                        clause_start:clause_end
                    ].strip()

                    if not clause_text:
                        raise ValueError(
                            f"Empty legal clause: "
                            f"{section_number}(1)({letter})"
                        )

                    # Include introductory context in each clause
                    complete_clause = (
                        f"{introduction} "
                        f"({letter}) {clause_text}"
                    )

                    chunks.append({
                        "text": complete_clause,
                        "reference": (
                            f"{section_number}(1)({letter})"
                        ),
                        "section": section_number,
                        "page_number": None
                    })

            # ------------------------------------------
            # Normal numbered subsections
            # ------------------------------------------

            else:

                if not subsection_text:
                    raise ValueError(
                        f"Empty legal subsection: {base_reference}"
                    )

                chunks.append({
                    "text": subsection_text,
                    "reference": base_reference,
                    "section": section_number,
                    "page_number": None
                })

    if not chunks:
        raise ValueError(
            "No legal document chunks were created."
        )

    # Ensure every legal reference is unique
    references = [
        chunk["reference"] for chunk in chunks
    ]

    if len(references) != len(set(references)):
        raise ValueError(
            "Duplicate legal references found in generated chunks."
        )

    print("\nLegal document chunking completed.")
    print("Total legal chunks:", len(chunks))

    return chunks


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

def get_embedding_model():

    global _model

    if _model is None:

        print("\nLoading embedding model...")

        _model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully.")

    return _model


# ==========================================
# 5. SEARCH LEGAL INFORMATION
# ==========================================

def search_legal_information(
    question,
    top_k=DEFAULT_TOP_K
):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

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

    print(
        "Retrieved legal documents:",
        len(legal_results)
    )

    for rank, result in enumerate(
        legal_results,
        start=1
    ):

        print(f"\nResult {rank}")

        print("Legal reference:", result["reference"])

        print(
            f"Similarity score: {result['score']:.3f}"
        )

        print("Legal text:", result["text"])

        print("-" * 60)

    return legal_results


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    question = input(
        "Ask a question about Norwegian employment law: "
    ).strip()

    if question:
        search_legal_information(question)
    else:
        print("Please enter a question.")
