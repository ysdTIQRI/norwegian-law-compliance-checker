from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample employment contract
contract_text = """
EMPLOYMENT CONTRACT

1. Employee Information

The employee name is John Smith.
The employee will work for ABC Company in Oslo, Norway.

2. Salary Information

The employee will receive a monthly salary of NOK 45,000.
Salary payments will be made on the last working day of each month.
Additional payments will be specified separately.

3. Working Hours

The employee will work 40 hours per week.
Normal working hours are from 9 AM to 5 PM.

4. Annual Leave

The employee is entitled to annual leave according to applicable Norwegian law.
Holiday pay will be calculated according to the relevant legal requirements.
"""

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Split the contract into chunks
chunks = text_splitter.split_text(contract_text)

# Display the results
print("\n========== DOCUMENT CHUNKING RESULTS ==========\n")

print("Total number of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):

    print(f"\n========== CHUNK {index} ==========\n")
    print(chunk)
    print("\nChunk length:", len(chunk))