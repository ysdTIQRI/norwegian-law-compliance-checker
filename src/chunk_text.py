
import pymupdf

# Open the PDF
pdf = pymupdf.open("data/sample.pdf")

# Extract text from all pages
full_text = ""

for page in pdf:
    full_text += page.get_text()

pdf.close()

# Split the text into chunks
chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

# Display the chunks
for number, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {number} ---")
    print(chunk)
