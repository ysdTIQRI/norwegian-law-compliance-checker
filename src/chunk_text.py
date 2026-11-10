
from pathlib import Path

import pymupdf
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_FILE = BASE_DIR / "data" / "employment_contract.pdf"

SUPPORTED_FORMATS = [".pdf", ".csv", ".xlsx", ".xls"]


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_pages(file_path):
    """
    Extract text from each PDF page.
    Preserve the original page number.
    """

    pages = []

    with pymupdf.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text()

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


# --------------------------------------------------
# CSV EXTRACTION
# --------------------------------------------------

def extract_csv_pages(file_path):
    """
    Extract CSV content and convert each row into readable text.
    Preserve CSV row numbers.
    """

    try:
        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path,
                encoding="cp1252"
            )

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")

    if df.empty:
        raise ValueError("The CSV file contains no data.")

    if len(df.columns) == 0:
        raise ValueError("No columns found in the CSV file.")

    pages = []

    for index, row in df.iterrows():

        row_text = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )

        if row_text.strip():

            pages.append({
                "page": f"CSV Row {index + 2}",
                "text": row_text
            })

    return pages


# --------------------------------------------------
# EXCEL EXTRACTION
# --------------------------------------------------

def extract_excel_pages(file_path):
    """
    Extract data from Excel worksheets.
    Preserve worksheet names and row numbers.

    Automatically close the Excel workbook after extraction
    to prevent temporary-file locking issues on Windows.
    """

    pages = []

    # Open the Excel workbook using a context manager.
    # The workbook will automatically close after processing.
    with pd.ExcelFile(file_path) as excel_file:

        for sheet_name in excel_file.sheet_names:

            df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name
            )

            if df.empty:
                continue

            for index, row in df.iterrows():

                row_text = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                    if pd.notna(value)
                )

                if row_text.strip():

                    pages.append({
                        "page": f"Sheet: {sheet_name}, Row: {index + 2}",
                        "text": row_text
                    })

    return pages


# --------------------------------------------------
# GENERAL DOCUMENT EXTRACTION
# --------------------------------------------------

def extract_pages(file_path=PDF_FILE):
    """
    Automatically detect the uploaded file format
    and extract its text.

    Supported formats:
    PDF, CSV, XLSX and XLS.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    if extension == ".pdf":

        pages = extract_pdf_pages(file_path)

    elif extension == ".csv":

        pages = extract_csv_pages(file_path)

    elif extension in [".xlsx", ".xls"]:

        pages = extract_excel_pages(file_path)

    else:

        pages = []

    if not pages:
        raise ValueError(
            "No readable text or data was found in the document."
        )

    return pages


# --------------------------------------------------
# CREATE CHUNKS USING LANGCHAIN
# --------------------------------------------------

def create_chunks(file_path=PDF_FILE, chunk_size=500, overlap=100):
    """
    Extract document text and split it using LangChain.

    Preserves the original page, CSV row or Excel
    worksheet and row information.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be between 0 and chunk_size - 1."
        )

    pages = extract_pages(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            if chunk.strip():

                chunks.append({
                    "page": page["page"],
                    "text": chunk
                })

    return chunks


# --------------------------------------------------
# TEST EXTRACTION
# --------------------------------------------------

if __name__ == "__main__":

    pages = extract_pages()

    print("Document extraction successful.")
    print("Number of pages/sections:", len(pages))

    chunks = create_chunks()

    print("\nNumber of chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\n--- Chunk {index} | "
            f"Location: {chunk['page']} ---"
        )

        print(chunk["text"])
