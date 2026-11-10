
import time
from pathlib import Path

from rag_pipeline import (
    PDF_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    create_chunks_from_pages,
    load_embedding_model,
    retrieve_contract_chunks,
    retrieve_legal_chunks
)

from chunk_text import extract_pages
from security_validation import validate_document


# --------------------------------------------------
# BENCHMARK CONFIGURATION
# --------------------------------------------------

TOP_K_CONTRACT = 3
TOP_K_LEGAL = 5

PAGE_MARKER = "\n\n<<<PAGE_BREAK_2026>>>\n\n"


# --------------------------------------------------
# TEST DATASET
# --------------------------------------------------

TEST_QUESTIONS = [
    {
        "question": (
            "According to the employment contract, "
            "what are the employee's normal working hours "
            "per week and daily schedule?"
        ),
        "expected_contract_text": "37.5 hours per week",
        "expected_legal_reference": "14-6(1)(j)"
    },
    {
        "question": (
            "What is the monthly salary, when is it paid, "
            "and where is the payment sent?"
        ),
        "expected_contract_text": "NOK 35,000 per month",
        "expected_legal_reference": "14-6(1)(i)"
    },
    {
        "question": (
            "What does the employment contract say about "
            "overtime arrangements and compensation?"
        ),
        "expected_contract_text": "Overtime should be agreed",
        "expected_legal_reference": "14-6(1)(m)"
    }
]


# --------------------------------------------------
# PREPARE CONTRACT DOCUMENT
# --------------------------------------------------

def prepare_contract_chunks():

    print("\nPreparing contract document...")

    pages = extract_pages(PDF_FILE)

    if not pages:
        raise ValueError("No text extracted from the PDF.")

    complete_text = PAGE_MARKER.join(
        page["text"] for page in pages
    )

    security_result = validate_document(complete_text)

    if not security_result["allowed"]:
        raise ValueError(
            "Document blocked by security validation."
        )

    masked_page_texts = security_result["masked_text"].split(
        PAGE_MARKER
    )

    if len(masked_page_texts) != len(pages):
        raise ValueError(
            "Could not restore PDF page structure after masking."
        )

    masked_pages = []

    for index, page in enumerate(pages):
        masked_pages.append({
            "page": page["page"],
            "text": masked_page_texts[index]
        })

    chunks = create_chunks_from_pages(
        masked_pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print("Contract pages:", len(pages))
    print("Contract chunks:", len(chunks))

    return chunks


# --------------------------------------------------
# CHECK CONTRACT RETRIEVAL ACCURACY
# --------------------------------------------------

def check_contract_retrieval(results, expected_text):

    for result in results:

        retrieved_text = result["chunk"]["text"].lower()

        if expected_text.lower() in retrieved_text:
            return True

    return False


# --------------------------------------------------
# CHECK LEGAL RETRIEVAL ACCURACY
# --------------------------------------------------

def check_legal_retrieval(results, expected_reference):

    for result in results:

        if result["reference"] == expected_reference:
            return True

    return False


# --------------------------------------------------
# RUN RETRIEVAL BENCHMARK
# --------------------------------------------------

def run_benchmark():

    print("\n========================================")
    print(" NORWEGIAN LAW CHECKER")
    print(" RETRIEVAL BENCHMARK")
    print("========================================")

    # Prepare masked contract chunks
    contract_chunks = prepare_contract_chunks()

    # Load embedding model once
    model = load_embedding_model()

    benchmark_results = []

    total_contract_time = 0
    total_legal_time = 0

    contract_hits = 0
    legal_hits = 0

    # Test each question
    for index, test in enumerate(TEST_QUESTIONS, start=1):

        question = test["question"]

        print(f"\n\n========== TEST {index} ==========")
        print("Question:", question)

        # Contract retrieval timing
        contract_start = time.perf_counter()

        contract_results = retrieve_contract_chunks(
            question,
            contract_chunks,
            model,
            TOP_K_CONTRACT
        )

        contract_time = (
            time.perf_counter() - contract_start
        ) * 1000

        # Legal retrieval timing
        legal_start = time.perf_counter()

        legal_results = retrieve_legal_chunks(
            question,
            model,
            TOP_K_LEGAL
        )

        legal_time = (
            time.perf_counter() - legal_start
        ) * 1000

        # Check retrieval accuracy
        contract_correct = check_contract_retrieval(
            contract_results,
            test["expected_contract_text"]
        )

        legal_correct = check_legal_retrieval(
            legal_results,
            test["expected_legal_reference"]
        )

        if contract_correct:
            contract_hits += 1

        if legal_correct:
            legal_hits += 1

        total_contract_time += contract_time
        total_legal_time += legal_time

        # Display individual results
        print("\nContract Retrieval:")

        for result in contract_results:
            print(
                "Page:", result["chunk"]["page"],
                "| Similarity:", round(result["score"], 3)
            )

        print("\nLegal Retrieval:")

        for result in legal_results:
            print(
                "Reference:", result["reference"],
                "| Similarity:", round(result["score"], 3)
            )

        print("\nExpected Contract Text:", test["expected_contract_text"])
        print("Contract Retrieval:", "PASS" if contract_correct else "FAIL")

        print("Expected Legal Reference:", test["expected_legal_reference"])
        print("Legal Retrieval:", "PASS" if legal_correct else "FAIL")

        print("Contract Retrieval Time:", round(contract_time, 2), "ms")
        print("Legal Retrieval Time:", round(legal_time, 2), "ms")

        benchmark_results.append({
            "test": index,
            "contract_correct": contract_correct,
            "legal_correct": legal_correct,
            "contract_time": contract_time,
            "legal_time": legal_time
        })

    # --------------------------------------------------
    # FINAL BENCHMARK SUMMARY
    # --------------------------------------------------

    total_tests = len(TEST_QUESTIONS)

    contract_accuracy = (
        contract_hits / total_tests
    ) * 100

    legal_accuracy = (
        legal_hits / total_tests
    ) * 100

    average_contract_time = (
        total_contract_time / total_tests
    )

    average_legal_time = (
        total_legal_time / total_tests
    )

    print("\n\n========================================")
    print(" FINAL BENCHMARK RESULTS")
    print("========================================")

    print(
        f"Contract Retrieval Accuracy: "
        f"{contract_hits}/{total_tests} "
        f"({contract_accuracy:.2f}%)"
    )

    print(
        f"Legal Retrieval Accuracy: "
        f"{legal_hits}/{total_tests} "
        f"({legal_accuracy:.2f}%)"
    )

    print(
        "Average Contract Retrieval Time:",
        round(average_contract_time, 2), "ms"
    )

    print(
        "Average Legal Retrieval Time:",
        round(average_legal_time, 2), "ms"
    )

    print(
        "Average Combined Retrieval Time:",
        round(
            average_contract_time + average_legal_time,
            2
        ),
        "ms"
    )

    print("========================================")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
