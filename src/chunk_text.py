from pathlib import Path
import pymupdf

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_FILE = BASE_DIR / "data" / "employment_contract.pdf"


def create_chunks(pdf_path=PDF_FILE, chunk_size=500, overlap=100):
    """
    Extract text from a PDF and divide it into overlapping chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of dictionaries containing page numbers and chunk text.
    """

    pdf_path = Path(pdf_path)

    # Check whether the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Validate chunk settings
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size - 1.")

    chunks = []

    # Open the PDF and extract text from each page
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            # Skip pages without text
            if not text.strip():
                continue

            # Create overlapping chunks
            step = chunk_size - overlap

            for i in range(0, len(text), step):
                chunk = text[i:i + chunk_size]

                if chunk.strip():
                    chunks.append({
                        "page": page_number,
                        "text": chunk
                    })

    return chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print("Number of chunks:", len(chunks))

    for number, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {number} | Page {chunk['page']} ---")
        print(chunk["text"])
