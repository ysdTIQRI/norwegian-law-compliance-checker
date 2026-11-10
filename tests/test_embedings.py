from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Load the embedding model
model = SentenceTransformer(MODEL_NAME)

# Sample legal text
text = "The employee's salary must be specified in the employment contract."

# Convert text into an embedding
embedding = model.encode(text)

# Display the results
print("Original text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nEmbedding dimensions:")
print(len(embedding))