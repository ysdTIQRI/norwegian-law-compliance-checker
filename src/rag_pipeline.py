import pymupdf
from sentence_transformers import SentenceTransformer, util

# STEP 1: Extract text from the PDF
pdf_path = "data/sample.pdf"

pdf = pymupdf.open(pdf_path)
pages = []

for page_number, page in enumerate(pdf, start=1):
    text = page.get_text()
    pages.append((page_number, text))

pdf.close()

full_text = "\n".join(text for _, text in pages)

if not full_text.strip():
    print("No text was found in the PDF.")
    raise SystemExit

print("PDF text extracted successfully.")

# STEP 2: Split the text into chunks
chunk_size = 500
chunks = []

for page_number, text in pages:
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append({
                "page": page_number,
                "text": chunk
            })

if not chunks:
    print("No text chunks were created.")
    raise SystemExit

print("Number of chunks:", len(chunks))

# STEP 3: Create embeddings
print("Loading the embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True
)

print("Embeddings created successfully.")

# STEP 4: Ask a question
query = input("\nAsk a question about the document: ").strip()

if not query:
    print("Please enter a question.")
    raise SystemExit

# STEP 5: Search for relevant chunks
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

similarities = util.cos_sim(
    query_embedding,
    chunk_embeddings
)[0]

number_of_results = min(3, len(chunks))

top_results = similarities.argsort(
    descending=True
)[:number_of_results]

# STEP 6: Display relevant text
print("\n--- Relevant Contract Information ---")

for index in top_results:
    chunk = chunks[index.item()]
    score = similarities[index.item()].item()

    print(f"\nPage: {chunk['page']}")
    print(f"Similarity score: {score:.3f}")
    print("Text:")
    print(chunk["text"])
    print("-" * 50)
