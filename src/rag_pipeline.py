from pathlib import Path

import time
import requests
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Support both direct execution and Streamlit package imports
try:
    from .chunk_text import extract_pages
    from .security_validation import validate_document
    from .vector_database import search_legal_documents

except ImportError:
    from chunk_text import extract_pages
    from security_validation import validate_document
    from vector_database import search_legal_documents


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FILE = BASE_DIR / "data" / "employment_contract.pdf"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K_CONTRACT = 5
TOP_K_LEGAL = 5


# --------------------------------------------------
# CREATE CHUNKS USING LANGCHAIN
# --------------------------------------------------

def create_chunks_from_pages(pages, chunk_size=500, overlap=100):
    """
    Create document chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Original PDF page numbers are preserved.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk overlap.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(page["text"])

        for chunk in page_chunks:

            if chunk.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk
                })

    return chunks


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():
    """
    Load the Sentence Transformers embedding model.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


# --------------------------------------------------
# RETRIEVE CONTRACT CHUNKS
# --------------------------------------------------

def retrieve_contract_chunks(question, chunks, model, top_k=3):
    """
    Retrieve the most relevant contract chunks
    based on semantic similarity.
    """

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

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
# RETRIEVE LEGAL INFORMATION FROM POSTGRESQL
# --------------------------------------------------

def retrieve_legal_chunks(question, model, top_k=5):
    """
    Retrieve relevant Norwegian legal sections
    from PostgreSQL using pgvector cosine similarity.
    """

    question_embedding = model.encode(
        question,
        convert_to_numpy=True
    )

    retrieved_chunks = search_legal_documents(
        question_embedding,
        top_k
    )

    if not retrieved_chunks:
        raise ValueError(
            "No legal documents found in PostgreSQL. "
            "Store the legal documents before running the pipeline."
        )

    return retrieved_chunks


# --------------------------------------------------
# CALL LOCAL QWEN MODEL USING OLLAMA
# --------------------------------------------------

def ask_local_llm(prompt):
    """
    Send the analysis prompt to the local Ollama model.
    """

    print("\nQwen is analysing the retrieved information...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 900,
                "num_ctx": 8192,
                "temperature": 0.1
            }
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["response"]


# --------------------------------------------------
# PREPARE ANALYSIS PROMPT
# --------------------------------------------------


def create_analysis_prompt(question, contract_results, legal_results):
    """
    Create a focused prompt for Norwegian employment
    contract analysis using retrieved evidence.
    """

    contract_context = "\n\n".join(
        f"""
Document location: {result["chunk"]["page"]}

Contract text:
{result["chunk"]["text"]}
"""
        for result in contract_results
    )

    legal_context = "\n\n".join(
        f"""
Legal reference: {result["reference"]}

Legal text:
{result["text"]}
"""
        for result in legal_results
    )

    return f"""
You are an AI assistant analysing a Norwegian employment contract.

Your task is to answer the user's question using ONLY
the supplied contract evidence and legal evidence.

USER QUESTION:
{question}


CONTRACT EVIDENCE:
{contract_context}


SUPPLIED NORWEGIAN LEGAL EVIDENCE:
{legal_context}


IMPORTANT ANALYSIS RULES:

1. Answer the user's actual question directly.

2. Use simple, clear and professional English.

3. Use only the supplied contract and legal evidence.

4. Never invent contract information or legal requirements.

5. Never contradict information explicitly stated in the contract.

6. Clearly distinguish example information from confirmed
   employment agreements.

7. Do not treat a fictional test document as a real agreement
   or automatically classify its fictional status as a violation.

8. Do not report information as missing if it is explicitly
   provided in the retrieved contract evidence.

9. If evidence is insufficient, clearly say
   "Insufficient information".

10. Never declare the entire contract legal or illegal.

11. Use only the legal references supplied above.

12. Similarity scores indicate retrieval relevance.
    They are not proof of legal compliance.

13. Discuss only topics relevant to the user's question.

14. Do not repeat the same finding in multiple sections.

15. Prioritise relevant contract evidence when answering
    questions about specific contract details.

16. Do not automatically classify missing information
    as a confirmed violation.

17. Distinguish between:
    - Information explicitly stated in the contract.
    - Information that is missing from the retrieved evidence.
    - Information that cannot be verified from the available evidence.

18. When identifying a potential compliance concern,
    explain exactly which contract information is missing
    or unclear and why the supplied legal provision is relevant.

19. Do not claim that the employer and employee have agreed
    on something unless the contract evidence establishes it.

20. Do not make assumptions about legal requirements
    that are not supported by the supplied legal text.

21. IMPORTANT LEGAL EVIDENCE RULE:
    If a retrieved legal provision explicitly addresses
    the user's question, acknowledge and explain that provision
    accurately.

    Never say that the supplied legal evidence contains no
    relevant provision when a relevant provision is present.

    Do not contradict, ignore or misrepresent the wording
    of a retrieved legal provision.

22. If a legal provision is relevant but the available
    contract evidence is incomplete, distinguish the legal
    requirement from the information missing in the contract.

23. Do not treat a similarity score as proof that a legal
    provision applies to the specific contract.

24. IMPORTANT CONTRACT INTERPRETATION RULE:

    Never describe a contract topic as completely absent
    when the retrieved contract evidence contains a relevant
    clause about that topic.

    Distinguish between:
    - No relevant clause exists.
    - A general clause exists, but specific details are missing.
    - The contract provides sufficient specific details.

    For example, if a contract mentions overtime but does not
    specify the overtime payment amount, state that a general
    overtime clause exists but the payment details are missing.

25. LEGAL PROVISION RELEVANCE:

    Explain each legal provision according to its actual wording.

    Do not describe a provision about daily and weekly working
    hours as a specific overtime payment requirement unless
    the supplied legal text supports that interpretation.

26. AVOID REPETITION:

    Section D must identify specific potential concerns.

    Section E must only explain limitations of the analysis,
    such as incomplete evidence or the need to verify current law.

    Never repeat the same finding in both sections.

27. A general clause is not the same as a complete
    contractual arrangement.

    If a relevant clause exists, acknowledge it.
    Identify only the specific details that are missing.

28. LEGAL PROVISION 14-6(1)(j):

    This provision concerns daily and weekly working
    hours and variable working-hour arrangements.

    Do not use it as evidence of a specific overtime
    payment or overtime calculation requirement.

29. LEGAL PROVISION 14-6(1)(m):

    Explain that the provision addresses arrangements
    for work exceeding agreed working hours, including
    payment.

    Do not add legal interpretations that are not
    supported by the supplied legal text.

30. SECTION E:

    Do not repeat missing contract information already
    discussed in section D.

    Only mention limitations affecting the reliability
    of the analysis.

31. LEGAL REFERENCE ACCURACY:

    Never assign a contract statement to a legal provision
    unless the supplied legal text actually addresses that topic.

32. COLLECTIVE PAY AGREEMENTS:

    Section 14-6(1)(n) concerns information about collective
    pay agreements regulating the employment relationship.

    Do not interpret this provision as the monthly salary
    payment date or payment interval.

33. SALARY REQUIREMENTS:

    Section 14-6(1)(i) concerns:
    - Pay applicable or agreed at commencement.
    - Supplements and other remuneration.
    - Method of payment.
    - Payment intervals.

    Do not declare full compliance based only on the
    presence of a salary amount.

34. COMPLIANCE CONCLUSIONS:

    Distinguish between:
    - Information present in the contract.
    - Information required by the supplied legal provision.
    - Whether the available evidence is sufficient to
      establish compliance.

    Do not declare that a provision is fully satisfied
    without sufficient evidence.

35. ILLUSTRATIVE CONTRACT INFORMATION:

    If the contract explicitly identifies a clause as
    illustrative, hypothetical or provided for testing,
    always describe it as illustrative information.

    Never present illustrative information as a confirmed
    employment agreement.

36. WORKING HOURS:

    When answering questions about normal working hours,
    focus on the stated daily and weekly working hours.

    Do not introduce overtime concerns unless the user
    specifically asks about overtime or it is directly
    relevant to the question.

37. LEGAL REFERENCE 14-6(1)(j):

    This provision concerns the duration and organisation
    of daily and weekly working hours.

    If working hours vary, it also addresses information
    about this variation and the basis for calculating
    when work will be performed.

    Do not claim that the provision confirms the legality
    of a particular working schedule.

38. LEGAL REFERENCE 14-6(1)(l):

    This provision concerns agreements about special
    working-hour arrangements.

    Do not apply it automatically to an ordinary
    working schedule.

39. RELEVANT LEGAL EVIDENCE ONLY:

    Never include a legal provision simply because it
    appears in the retrieved results.

    Include only provisions whose actual wording is
    relevant to the user's question.

40. NO RELEVANT LEGAL PROVISION:

    If the retrieved legal provisions do not address
    the user's question, explicitly state:

    "Insufficient relevant legal evidence was retrieved."

    Do not invent or substitute unrelated legal provisions.

41. VERIFY CONTRACT EVIDENCE:

    Before identifying missing information, check all
    retrieved contract evidence.

    Do not report information as missing if it appears
    anywhere in the supplied evidence.

42. EXCLUDE TEST INSTRUCTIONS:

    Do not treat example questions, testing instructions,
    placeholders or document-processing instructions
    as actual employment contract terms.

43. PERSONAL DATA QUESTIONS:

    Distinguish between:
    - Confidentiality obligations.
    - Personal data processing information.
    - Employee data-protection rights.

    Do not claim that a general confidentiality clause
    establishes compliance with personal-data legislation.

44. NOTICE PERIOD:

    Section 14-6(1)(h) concerns applicable notice periods
    for employees and employers and termination procedures.

    Explain this provision according to its supplied wording.

45. TEMPORARY EMPLOYMENT:

    Section 14-6(1)(e) concerns temporary employment,
    expected duration and the basis for appointment.

    Do not automatically include this provision when
    the contract describes an ongoing employment relationship.

46. NOTICE PERIOD ACCURACY:

    Distinguish between an illustrative notice period
    and an actual agreed employment term.

    Do not confirm that an illustrative notice period
    applies to a real employment relationship.

47. CONTRACT COMPLETENESS:

    Do not report information as missing when it is
    present in the retrieved contract evidence.

    Instead, explain whether the information is
    illustrative, incomplete or unverified.



RESPONSE FORMAT:

A. DIRECT ANSWER

Answer the user's question in no more than 2 sentences.

State only the main finding.
Do not explain the evidence or legal provisions here.


B. CONTRACT EVIDENCE

Provide the exact relevant sentence or sentences
from the retrieved contract text.

Include the original PDF page number.

Do not paraphrase the evidence.
Do not explain the compliance implications.


C. RELEVANT LEGAL PROVISIONS

Explain the relevant legal provisions separately.

For each provision:
- Give the legal reference.
- Explain its actual wording in simple English.
- Explain its relevance to the user's question.

Do not add legal requirements that are not present
in the supplied legal text.

In particular:
- 14-6(1)(m) concerns arrangements for work exceeding
  agreed working hours, including payment.
- 14-6(1)(j) concerns daily and weekly working hours
  and variations in working hours.

Do not claim that either provision requires an
agreement or confirmation unless that requirement
is explicitly supported by the supplied legal text.


D. POTENTIAL COMPLIANCE CONCERNS

Identify only specific concerns supported by
the contract and legal evidence.

For each concern, explain:
- What information is missing or unclear.
- Which legal provision is relevant.

Do not repeat the direct answer or contract evidence.

If no specific concern can be established,
state: "No specific compliance concern can be
confirmed from the available evidence."


E. LIMITATIONS

Use a maximum of 2 sentences.

Mention only limitations affecting the reliability
of the analysis.

Do not repeat findings, missing contract details,
or compliance concerns already explained in D.

Mention that the analysis is preliminary and must
be verified against current official Norwegian law.

This prototype does not provide legal advice.
"""




# --------------------------------------------------
# MAIN RAG PIPELINE - STREAMLIT AND PYTHON
# --------------------------------------------------

def run_analysis(document_path, question):
    """
    Run the complete RAG pipeline for PDF, CSV or Excel documents.

    Returns:
        Analysis
        Security status
        PII detection status
        Contract retrieval results
        Legal retrieval results
    """

    document_path = Path(document_path)

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    supported_formats = {".pdf", ".csv", ".xlsx", ".xls"}
    document_extension = document_path.suffix.lower()

    if document_extension not in supported_formats:
        raise ValueError(
            "Unsupported document format. Supported formats: "
            "PDF, CSV, XLSX and XLS."
        )

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    question = question.strip()

    # STEP 1: Extract document content

    print("\n========== DOCUMENT PROCESSING ==========\n")

    pages = extract_pages(document_path)

    if not pages:
        raise ValueError("No readable text or data was extracted from the document.")

    print("Document loaded successfully.")
    print("Document format:", document_extension.upper().replace(".", ""))
    print("Number of pages/sections/rows:", len(pages))

    # STEP 2: Prepare complete contract text

    page_marker = "\n\n<<<PAGE_BREAK_2026>>>\n\n"

    complete_contract_text = page_marker.join(
        page["text"] for page in pages
    )

    # STEP 3: Validate complete contract and mask PII

    print("\n========== DOCUMENT SECURITY ==========\n")

    security_result = validate_document(
        complete_contract_text
    )

    if not security_result["allowed"]:

        return {
            "allowed": False,
            "message": "Document blocked by security validation.",
            "analysis": None,
            "page_count": len(pages),
            "document_location_count": len(pages),
            "document_type": document_extension.lstrip("."),
            "chunk_count": 0,
            "pii_detected": security_result["pii_detected"],
            "masking_completed": security_result["masking_completed"],
            "contract_results": [],
            "legal_results": []
        }

    masked_contract_text = security_result["masked_text"]

    # STEP 4: Restore original document section structure

    masked_page_texts = masked_contract_text.split(page_marker)

    if len(masked_page_texts) != len(pages):
        raise ValueError(
            "Could not restore the original document structure "
            "after PII masking."
        )

    masked_pages = []

    for index, page in enumerate(pages):

        masked_pages.append({
            "page": page["page"],
            "text": masked_page_texts[index]
        })

    print("\nDocument passed security validation.")
    print("PII masking completed.")
    print("Original document locations preserved.")

    # STEP 5: Create contract chunks using LangChain

    print("\n========== CONTRACT CHUNKING ==========\n")

    contract_chunks = create_chunks_from_pages(
        masked_pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    if not contract_chunks:
        raise ValueError("No contract chunks were created.")

    print("Number of masked contract chunks:", len(contract_chunks))
    print("Chunking method: LangChain RecursiveCharacterTextSplitter")

    # STEP 6: Load embedding model

    embedding_model = load_embedding_model()

    # STEP 7: Retrieve relevant contract information

    print("\n========== CONTRACT RETRIEVAL ==========\n")

    contract_start_time = time.perf_counter()

    contract_results = retrieve_contract_chunks(
        question,
        contract_chunks,
        embedding_model,
        TOP_K_CONTRACT
    )

    contract_retrieval_time = (
        time.perf_counter() - contract_start_time
    ) * 1000

    print(
        f"Contract retrieval time: "
        f"{contract_retrieval_time:.2f} ms"
    )

    print("Retrieved contract chunks:", len(contract_results))

    # STEP 8: Retrieve relevant legal information

    print("\n========== LEGAL RETRIEVAL ==========\n")

    legal_start_time = time.perf_counter()

    legal_results = retrieve_legal_chunks(
        question,
        embedding_model,
        TOP_K_LEGAL
    )

    legal_retrieval_time = (
        time.perf_counter() - legal_start_time
    ) * 1000

    print(
        f"Legal retrieval time: "
        f"{legal_retrieval_time:.2f} ms"
    )

    print("Retrieved legal sections:", len(legal_results))

    # STEP 9: Prepare question-focused analysis prompt

    prompt = create_analysis_prompt(
        question,
        contract_results,
        legal_results
    )

    # STEP 10: Analyse using local Qwen model

    analysis = ask_local_llm(prompt)

    print("\n========== ANALYSIS COMPLETED ==========\n")

    # STEP 11: Return all results

    return {
        "allowed": True,
        "message": "Document passed security validation.",
        "analysis": analysis,
        "page_count": len(pages),
        "document_location_count": len(pages),
        "document_type": document_extension.lstrip("."),
        "chunk_count": len(contract_chunks),

        "pii_detected": security_result["pii_detected"],
        "masking_completed": security_result["masking_completed"],

        "contract_results": contract_results,
        "legal_results": legal_results,

        # Retrieval performance
        "contract_retrieval_time": contract_retrieval_time,
        "legal_retrieval_time": legal_retrieval_time,
        "total_retrieval_time": (
            contract_retrieval_time + legal_retrieval_time
        ),

        # Verified masked document content
        "masked_document_text": masked_contract_text
    }


# --------------------------------------------------
# COMMAND-LINE VERSION - POWERSHELL
# --------------------------------------------------

def answer_question(document_path=PDF_FILE):

    question = input(
        "\nAsk a question or request a compliance check: "
    ).strip()

    try:

        result = run_analysis(document_path, question)

        if not result["allowed"]:
            print("\nBLOCKED:", result["message"])
            return

        print("\n========== COMPLIANCE ANALYSIS ==========\n")

        print(result["analysis"])

        print("\n========== RETRIEVED DOCUMENT SOURCES ==========\n")

        for number, item in enumerate(
            result["contract_results"],
            start=1
        ):

            chunk = item["chunk"]

            print(f"Result {number}")
            print("Document location:", chunk["page"])
            print("Similarity:", round(item["score"], 3))
            print(chunk["text"])
            print("-" * 60)

        print("\n========== RETRIEVED LEGAL SOURCES ==========\n")

        for number, item in enumerate(
            result["legal_results"],
            start=1
        ):

            print(f"Result {number}")
            print("Legal reference:", item["reference"])
            print("Similarity:", round(item["score"], 3))
            print(item["text"])
            print("-" * 60)

    except requests.exceptions.ConnectionError:

        print("\nERROR: Cannot connect to Ollama.")
        print("Please make sure Ollama is running.")

    except requests.exceptions.HTTPError as error:

        print("\nOllama API error:", error)

    except Exception as error:

        print("\nAn error occurred:", error)


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    answer_question()
