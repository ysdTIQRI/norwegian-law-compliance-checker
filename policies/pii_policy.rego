package pii_policy

import rego.v1

# Default: Block document processing
default decision := {
    "allow": false,
    "message": "Document processing blocked. PII masking is required."
}

# Allow processing when no PII is detected
decision := {
    "allow": true,
    "message": "No PII detected. Document processing allowed."
} if {
    input.pii_detected == false
}

# Allow processing when PII is detected and masking is completed
decision := {
    "allow": true,
    "message": "PII detected and successfully masked. Processing allowed."
} if {
    input.pii_detected == true
    input.masking_completed == true
}