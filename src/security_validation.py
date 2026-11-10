
import json
import subprocess
import tempfile
from pathlib import Path

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

try:
    from .norwegian_pii_recognizers import (
        NorwegianFodselsnummerRecognizer
    )
except ImportError:
    from norwegian_pii_recognizers import (
        NorwegianFodselsnummerRecognizer
    )


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

OPA_PATH = BASE_DIR / "opa.exe"
POLICY_PATH = BASE_DIR / "policies" / "pii_policy.rego"

PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "NORWEGIAN_FODSELSNUMMER"
]

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Register Norwegian fødselsnummer recognizer
analyzer.registry.add_recognizer(
    NorwegianFodselsnummerRecognizer()
)

# ==========================================
# 2. DOCUMENT SECURITY VALIDATION
# ==========================================

def validate_document(text):
    """
    Detect and mask PII in extracted document text.

    Verify that all targeted PII is removed before
    allowing the document to proceed through OPA.

    Supports:
    - PDF
    - CSV
    - Excel
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text is empty or invalid.")

    if not OPA_PATH.exists():
        raise FileNotFoundError(
            f"OPA executable not found: {OPA_PATH}"
        )

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"OPA policy not found: {POLICY_PATH}"
        )

    # ------------------------------------------
    # STEP 1: DETECT PII
    # ------------------------------------------

    print("\n========== PII VALIDATION ==========\n")

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=PII_ENTITIES
    )

    pii_detected = len(results) > 0

    if pii_detected:

        print("PII detected.")

        for result in results:

            print(
                f"Type: {result.entity_type} | "
                f"Confidence: {result.score:.2f} | "
                "Value: [REDACTED]"
            )

    else:

        print("No PII detected.")

    # ------------------------------------------
    # STEP 2: MASK DETECTED PII
    # ------------------------------------------

    operators = {
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
        ),

       "NORWEGIAN_FODSELSNUMMER": OperatorConfig(
            "replace",
            {"new_value": "[REDACTED]"}
)
    }

    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )

    masked_text = anonymized_result.text

    # ------------------------------------------
    # STEP 3: VERIFY MASKING
    # ------------------------------------------

    print("\n========== PII MASKING VERIFICATION ==========\n")

    if not pii_detected:

        masking_completed = True

        print("No PII detected. Masking not required.")

    else:

        # Analyse the masked document again.
        # Check whether any targeted PII remains.

        remaining_results = analyzer.analyze(
            text=masked_text,
            language="en",
            entities=PII_ENTITIES
        )

        masking_completed = len(remaining_results) == 0

        if masking_completed:

            print("PII masking verification successful.")
            print("No targeted PII remains detectable.")

        else:

            print("PII masking verification FAILED.")

            print(
                "Remaining detected PII entities:",
                len(remaining_results)
            )

            for result in remaining_results:

                print(
                    f"Remaining type: {result.entity_type} | "
                    f"Confidence: {result.score:.2f} | "
                    "Value: [REDACTED]"
                )

            print(
                "Document will be blocked by OPA."
            )

    # ------------------------------------------
    # STEP 4: PREPARE OPA INPUT
    # ------------------------------------------

    policy_input = {
        "pii_detected": pii_detected,
        "masking_completed": masking_completed
    }

    temp_file_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False
        ) as temp_file:

            json.dump(policy_input, temp_file)

            temp_file_path = temp_file.name

        # ------------------------------------------
        # STEP 5: ASK OPA FOR SECURITY DECISION
        # ------------------------------------------

        command = [
            str(OPA_PATH),
            "eval",
            "--format=json",
            "--data",
            str(POLICY_PATH),
            "--input",
            temp_file_path,
            "data.pii_policy.decision"
        ]

        response = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        opa_result = json.loads(response.stdout)

        result_items = opa_result.get("result", [])

        if not result_items:
            raise RuntimeError(
                "OPA returned an empty security decision."
            )

        expressions = result_items[0].get(
            "expressions", []
        )

        if not expressions:
            raise RuntimeError(
                "OPA did not return a decision expression."
            )

        decision = expressions[0].get("value")

        if not isinstance(decision, dict):
            raise RuntimeError(
                "OPA returned an invalid decision."
            )

        if not isinstance(decision.get("allow"), bool):
            raise RuntimeError(
                "OPA decision does not contain a valid allow value."
            )

        if not isinstance(decision.get("message"), str):
            raise RuntimeError(
                "OPA decision does not contain a valid message."
            )

    except subprocess.TimeoutExpired as error:

        raise RuntimeError(
            "OPA security validation timed out."
        ) from error

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            f"OPA execution failed: {error.stderr}"
        ) from error

    finally:

        if temp_file_path:
            Path(temp_file_path).unlink(missing_ok=True)

    # ------------------------------------------
    # STEP 6: DISPLAY OPA SECURITY DECISION
    # ------------------------------------------

    print("\n========== OPA SECURITY DECISION ==========\n")

    print("Allow:", decision["allow"])
    print("Message:", decision["message"])

    # ------------------------------------------
    # STEP 7: RETURN SECURITY RESULTS
    # ------------------------------------------

    return {
        "pii_detected": pii_detected,
        "masking_completed": masking_completed,
        "allowed": decision["allow"],
        "masked_text": masked_text
    }


# ==========================================
# 3. TEST SECURITY VALIDATION
# ==========================================

if __name__ == "__main__":

    sample_contract = """
    EMPLOYMENT CONTRACT

    Employee Name: John Smith
    Email: john.smith@example.com
    Phone: +47 912 34 567

    Monthly salary: NOK 45,000.
    """

    result = validate_document(sample_contract)

    if result["allowed"]:

        print(
            "\nSUCCESS: Document is allowed "
            "for further processing."
        )

        print("\nMasked document:")
        print(result["masked_text"])

    else:

        print(
            "\nBLOCKED: Document cannot be processed."
        )
