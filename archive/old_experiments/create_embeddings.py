
import pymupdf
from sentence_transformers import SentenceTransformer

# Open the PDF
pdf = pymupdf.open("data/sample.pdf")

# Extract all text
full_text = ""

for page in pdf:
    full_text += page.get_text()

pdf.close()

# Split text into chunks
chunk_size = 500
chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks into embeddings
embeddings = model.encode(chunks)

# Display the results
print("Number of chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])
