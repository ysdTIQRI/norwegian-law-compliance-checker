import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, util

from chunk_text import create_chunks
from legal_search import load_legal_text, create_legal_chunks

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FILE = BASE_DIR / "data" / "employment_contract.pdf"

MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

TOP_K_CONTRACT = 3
TOP_K_LEGAL = 3

def retrieve_relevant_chunks(question, chunks, model, top_k=3):
    """
    Find the chunks that are most relevant to the question.

    Each chunk can be either:
    - A dictionary containing page and text (contract chunks).
    - A string containing legal text (legal chunks).
    """

    if not chunks:
        return []

    # Get the text from each chunk
    if isinstance(chunks[0], dict):
        texts = [chunk["text"] for chunk in chunks]
    else:
        texts = chunks

    # Create embeddings for the chunks
    chunk_embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )

    # Create an embedding for the question
    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Compare the question with each chunk
    similarities = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # Select the most relevant chunks
    number_to_return = min(top_k, len(chunks))

    top_indices = similarities.argsort(
        descending=True
    )[:number_to_return]

    results = []

    for index in top_indices:
        position = index.item()

        results.append({
            "chunk": chunks[position],
            "score": similarities[position].item()
        })

    return results


# --------------------------------------------------
# Main RAG pipeline
# --------------------------------------------------

def answer_question(pdf_path=PDF_FILE):

    # Step 1: Load the API key
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please check your .env file."
        )

    # Step 2: Check the employment contract
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Step 3: Extract and chunk the employment contract
    print("\n========== DOCUMENT PROCESSING ==========\n")

    contract_chunks = create_chunks(pdf_path)

    if not contract_chunks:
        print("No text was found in the employment contract.")
        return

    print("Employment contract loaded successfully.")
    print("Number of contract chunks:", len(contract_chunks))

    # Step 4: Load the Norwegian legal text
    print("\n========== LEGAL TEXT PROCESSING ==========\n")

    legal_text = load_legal_text()

    legal_chunks = create_legal_chunks(legal_text)

    if not legal_chunks:
        print("No legal text chunks were found.")
        return

    print("Norwegian legal text loaded successfully.")
    print("Number of legal chunks:", len(legal_chunks))

    # Step 5: Load the embedding model
    print("\nLoading the embedding model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    # Step 6: Ask the user a question
    question = input(
        "\nAsk a question or request a contract compliance check: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    # Step 7: Search the employment contract
    print("\nSearching the employment contract...")

    contract_results = retrieve_relevant_chunks(
        question=question,
        chunks=contract_chunks,
        model=embedding_model,
        top_k=TOP_K_CONTRACT
    )

    # Step 8: Search Norwegian legal provisions
    print("Searching Norwegian legal provisions...")

    legal_results = retrieve_relevant_chunks(
        question=question,
        chunks=legal_chunks,
        model=embedding_model,
        top_k=TOP_K_LEGAL
    )

    # Step 9: Prepare the contract context
    contract_context_parts = []

    for result in contract_results:
        chunk = result["chunk"]

        contract_context_parts.append(
            f"Contract page: {chunk['page']}\n"
            f"Similarity score: {result['score']:.3f}\n"
            f"Text:\n{chunk['text']}"
        )

    contract_context = "\n\n".join(contract_context_parts)

    # Step 10: Prepare the legal context
    legal_context_parts = []

    for result in legal_results:
        chunk = result["chunk"]

        legal_context_parts.append(
            f"Similarity score: {result['score']:.3f}\n"
            f"Legal text:\n{chunk}"
        )

    legal_context = "\n\n".join(legal_context_parts)

    print("\nRelevant contract and legal information retrieved.")

    # Step 11: Prepare the prompt for Gemini
    prompt = f"""
You are an assistant for a Norwegian Law Compliance Checker prototype.

Your task is to compare the employment contract with the Norwegian
legal provisions supplied below.

USER QUESTION:
{question}

EMPLOYMENT CONTRACT INFORMATION:
{contract_context}

NORWEGIAN LEGAL INFORMATION:
{legal_context}

INSTRUCTIONS:

1. Answer in simple, clear English.

2. Use only the supplied contract information and legal information
   when making factual claims about this contract and the law.

3. Compare relevant contract clauses with the supplied legal provisions.

4. Identify possible compliance concerns only when the supplied
   legal text supports the concern.

5. Explain why each potential issue may need further review.

6. Reference the relevant contract page numbers.

7. Reference the legal section numbers exactly as they appear
   in the supplied legal text.

8. If the legal text does not provide enough information to decide
   whether a clause complies with the law, clearly say so.

9. Do not invent legal requirements, section numbers, or contract terms.

10. Do not assume that a missing detail in the retrieved contract
    chunks is absent from the complete contract.

11. A similarity score indicates relevance to the question. It does
    not prove that a legal provision applies.

12. Do not declare the entire contract legal or illegal based only
    on the retrieved information.

13. Explain that this is a prototype and that its findings need
    verification against the current official Norwegian legal sources.

Use this response structure:

A. Summary
Briefly explain what you found.

B. Contract information
Describe the relevant contract clauses and page numbers.

C. Relevant Norwegian legal provisions
Explain the relevant legal text and section numbers.

D. Potential compliance concerns
For each concern, explain:
- What the contract says.
- What the supplied legal text says.
- Why the clause may need further review.
- Which contract page and legal section are relevant.

If no potential concern can be established from the supplied
information, say that the available information is insufficient
to identify one. Do not automatically conclude that the contract
complies with Norwegian law.

E. Limitations
Identify missing information or legal sources that should be checked.
"""

    # Step 12: Ask Gemini to analyse the information
    print("\nGemini is analysing the contract and legal provisions...")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    # Step 13: Display the answer
    print("\n========== COMPLIANCE ANALYSIS ==========\n")

    print(
        response.text
        or "Gemini did not return a text answer."
    )

    # Step 14: Display the retrieved contract sources
    print("\n========== RETRIEVED CONTRACT SOURCES ==========\n")

    for number, result in enumerate(contract_results, start=1):
        chunk = result["chunk"]

        print(f"Result {number}")
        print(f"Page: {chunk['page']}")
        print(f"Similarity score: {result['score']:.3f}")
        print(chunk["text"])
        print("-" * 60)

    # Step 15: Display the retrieved legal sources
    print("\n========== RETRIEVED LEGAL SOURCES ==========\n")

    for number, result in enumerate(legal_results, start=1):
        print(f"Result {number}")
        print(f"Similarity score: {result['score']:.3f}")
        print(result["chunk"])
        print("-" * 60)


# --------------------------------------------------
# Run the program
# --------------------------------------------------

if __name__ == "__main__":
    answer_question()