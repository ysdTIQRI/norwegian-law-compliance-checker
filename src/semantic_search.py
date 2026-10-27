
import pymupdf
from sentence_transformers import SentenceTransformer, util

# 1. Open the PDF
pdf = pymupdf.open("sample.pdf")

# 2. Extract all text
full_text = ""

for page in pdf:
    full_text += page.get_text()

pdf.close()

# 3. Split the text into chunks
chunk_size = 500
chunks = []

for i in range(0, len(full_text), chunk_size):
    chunks.append(full_text[i:i + chunk_size])

# 4. Load the embedding modelgit
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Convert document chunks into embeddings
chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)

# 6. Enter a search query
query = input("What do you want to search for? ")

# 7. Convert the query into an embedding
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

# 8. Compare the query with all chunks
similarities = util.cos_sim(
    query_embedding,
    chunk_embeddings
)[0]

# 9. Get the three most relevant chunks
top_results = similarities.argsort(
    descending=True
)[:3]

# 10. Display the results
for index in top_results:
    print("\n--- Relevant Text ---")
    print("Similarity:", round(similarities[index].item(), 3))
    print(chunks[index])
