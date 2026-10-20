from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Create the tools
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Example contract text using fake details
text = """
Employee name: John Smith.
Email: john.smith@example.com.
Phone number: 212-555-1234.
The employee receives a monthly salary.
"""

# Detect personal information
results = analyzer.analyze(
    text=text,
    language="en"
)

# Replace detected personal information
anonymized_result = anonymizer.anonymize(
    text=text,
    analyzer_results=results
)

# Display the results
print("Original text:")
print(text)

print("\nMasked text:")
print(anonymized_result.text)