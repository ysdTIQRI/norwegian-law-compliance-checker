from pathlib import Path
import re

from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent.parent

LEGAL_FILE = (
    BASE_DIR
    / "legal_data"
    / "working_environment_act.txt"
)

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

def load_legal_text():
    """Read the Norwegian legal text file."""

    if not LEGAL_FILE.exists():
        raise FileNotFoundError(
            f"Legal file not found: {LEGAL_FILE}"
        )

    text = LEGAL_FILE.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("The legal text file is empty.")

    return text

def create_legal_chunks(text):
    """
    Split legal text into smaller chunks.

    Each chunk keeps its legal reference, such as:
    14-5(1)
    14-6(1)(g)
    14-6(1)(h)
    """

    heading_pattern = re.compile(
        r"(?mi)^\s*(?:section\s+)?(\d+-\d+)\s*[-–—]\s*(.+?)\s*$"
    )

    headings = list(heading_pattern.finditer(text))

    if not headings:
        raise ValueError(
            "No legal section headings were found. "
            "Check the heading format in working_environment_act.txt."
        )

    chunks = []

    for index, heading in enumerate(headings):

        section_number = heading.group(1)

        start = heading.end()

        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        lines = section_text.splitlines()

        current_reference = None
        current_lines = []

        def save_chunk():

            if current_reference and current_lines:

                chunk_text = "\n".join(current_lines).strip()

                if chunk_text:
                    chunks.append({
                        "section": section_number,
                        "reference": current_reference,
                        "text": chunk_text
                    })

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Identify numbered subsections: (1), (2), (3)
            number_match = re.match(
                r"^\((\d+)\)\s*(.*)",
                line
            )

            # Identify lettered clauses: (a), (b), (c)
            letter_match = re.match(
                r"^\(([a-q])\)\s*(.*)",
                line,
                re.IGNORECASE
            )

            # Handle numbered subsections
            if number_match:

                save_chunk()

                subsection_number = number_match.group(1)

                current_reference = (
                    f"{section_number}({subsection_number})"
                )

                current_lines = []

                remaining_text = number_match.group(2).strip()

                if remaining_text:
                    current_lines.append(remaining_text)

            # Handle lettered clauses inside Section 14-6(1)
            elif (
                section_number == "14-6"
                and current_reference is not None
                and current_reference.startswith("14-6(1)")
                and letter_match
            ):

                save_chunk()

                clause_letter = letter_match.group(1).lower()

                current_reference = (
                    f"14-6(1)({clause_letter})"
                )

                current_lines = []

                remaining_text = letter_match.group(2).strip()

                if remaining_text:
                    current_lines.append(remaining_text)

            else:

                if current_reference:
                    current_lines.append(line)

        # Save the last chunk in the section
        save_chunk()

    return chunks

# --------------------------------------------------
# Step 3: Search for relevant legal provisions
# --------------------------------------------------

def search_legal_information(question, top_k=DEFAULT_TOP_K):
    """Find the most relevant legal chunks."""

    if not question.strip():
        raise ValueError("The question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Load and split legal text
    legal_text = load_legal_text()

    legal_chunks = create_legal_chunks(legal_text)

    if not legal_chunks:
        print("No legal sections were found.")
        return []

    print("Legal text loaded successfully.")
    print("Number of legal chunks:", len(legal_chunks))

    # Load embedding model
    print("\nLoading the embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Create embeddings for legal chunks
    legal_texts = [
        chunk["text"]
        for chunk in legal_chunks
    ]

    legal_embeddings = model.encode(
        legal_texts,
        convert_to_tensor=True
    )

    print("Legal embeddings created successfully.")

    # Create question embedding
    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Calculate semantic similarity
    similarities = util.cos_sim(
        question_embedding,
        legal_embeddings
    )[0]

    # Select the top results
    number_to_return = min(
        top_k,
        len(legal_chunks)
    )

    top_indices = similarities.argsort(
        descending=True
    )[:number_to_return]

    # Display results
    results = []

    print("\n========== LEGAL SEARCH RESULTS ==========\n")

    for rank, index in enumerate(top_indices, start=1):
        position = index.item()

        chunk = legal_chunks[position]
        score = similarities[position].item()

        result = {
            "section": chunk["section"],
            "reference": chunk["reference"],
            "text": chunk["text"],
            "score": score
        }

        results.append(result)

        print(f"Result {rank}")
        print(f"Legal section: {result['section']}")
        print(f"Legal reference: {result['reference']}")
        print(f"Similarity score: {score:.3f}")
        print(result["text"])
        print("-" * 60)

    return results


# --------------------------------------------------
# Step 4: Run the search
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Ask a question about Norwegian employment law: "
    ).strip()

    if question:
        search_legal_information(
            question,
            top_k=5
        )
    else:
        print("Please enter a question.")