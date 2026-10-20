
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Step 1: Create the Presidio tools
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Step 2: Read data from the CSV file
df = pd.read_csv("pii_data.csv", dtype=str)

print("Original PII Data:")
print(df)

# Step 3: Define a function to detect and mask PII
def mask_pii(value, column_name):
    if pd.isna(value) or not value.strip():
        return value

    # Add context to help Presidio identify the data type
    text = f"{column_name}: {value}"

    # Detect personal information
    results = analyzer.analyze(
        text=text,
        language="en"
    )

    # Mask detected personal information
    masked_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    # Remove the column label from the result
    prefix = f"{column_name}: "
    masked_text = masked_result.text

    if masked_text.startswith(prefix):
        masked_text = masked_text[len(prefix):]

    return masked_text

# Step 4: Apply masking to every column
for column in df.columns:
    df[column] = df[column].apply(
        lambda value: mask_pii(value, column)
    )

# Step 5: Save the masked data to a new CSV file
df.to_csv("masked_pii_data.csv", index=False)

print("\nMasked PII Data:")
print(df)

print("\nPII masking completed successfully!")
print("Output file: masked_pii_data.csv")
