from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize Presidio
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Sample employment contract
sample_text = """
EMPLOYMENT CONTRACT

Employee Name: John Smith
Email: john.smith@example.com
Phone: +47 912 34 567
Address: Oslo, Norway

The employee will receive a monthly salary of NOK 45,000.
"""

# Detect only the required PII entities
results = analyzer.analyze(
    text=sample_text,
    language="en",
    entities=[
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER"
    ]
)

print("\n========== PII VALIDATION ==========\n")

if results:
    print("PII detected!\n")

    for result in results:
        detected_value = sample_text[result.start:result.end]

        print(f"Type: {result.entity_type}")
        print(f"Value: {detected_value}")
        print(f"Confidence: {result.score:.2f}")
        print("-------------------------")

else:
    print("No PII detected.")

# Mask detected personal information
anonymized_result = anonymizer.anonymize(
    text=sample_text,
    analyzer_results=results,
    operators={
        "PERSON": OperatorConfig(
            "replace",
            {"new_value": "<PERSON>"}
        ),
        "EMAIL_ADDRESS": OperatorConfig(
            "replace",
            {"new_value": "<EMAIL_ADDRESS>"}
        ),
        "PHONE_NUMBER": OperatorConfig(
            "replace",
            {"new_value": "<PHONE_NUMBER>"}
        )
    }
)

print("\n========== MASKED DOCUMENT ==========\n")
print(anonymized_result.text)