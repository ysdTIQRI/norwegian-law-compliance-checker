import pandas as pd
from cryptography.fernet import Fernet
from pathlib import Path

# Get the project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SECRETS_DIR = BASE_DIR / "secrets"

# File paths
INPUT_FILE = DATA_DIR / "pii_data.csv"
OUTPUT_FILE = DATA_DIR / "encrypted_pii_data.csv"
KEY_FILE = SECRETS_DIR / "secret.key"


# STEP 1: Create the secrets folder if it does not exist
SECRETS_DIR.mkdir(parents=True, exist_ok=True)


# STEP 2: Generate an encryption key only if it does not exist
if not KEY_FILE.exists():
    key = Fernet.generate_key()

    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)

    print("Encryption key created successfully!")

else:
    print("Existing encryption key found.")


# STEP 3: Load the encryption key
with open(KEY_FILE, "rb") as key_file:
    key = key_file.read()

cipher = Fernet(key)


# STEP 4: Check whether the input CSV exists
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input CSV file not found: {INPUT_FILE}"
    )


# STEP 5: Read the PII data
df = pd.read_csv(INPUT_FILE, dtype=str)

print("\nOriginal PII Data:")
print(df)


# STEP 6: Select sensitive columns
sensitive_columns = [
    "FullName",
    "Email",
    "Phone",
    "Address",
    "NIC"
]

# Check that all required columns exist
missing_columns = [
    column for column in sensitive_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in the CSV file: {missing_columns}"
    )


# STEP 7: Encrypt the sensitive data
for column in sensitive_columns:
    df[column] = df[column].apply(
        lambda value: cipher.encrypt(value.encode("utf-8")).decode("utf-8")
        if pd.notna(value) and value != ""
        else value
    )


# STEP 8: Save the encrypted data
df.to_csv(OUTPUT_FILE, index=False)

print("\nEncrypted PII Data:")
print(df)

print("\nEncryption completed successfully!")
print(f"Encrypted file: {OUTPUT_FILE}")
print(f"Key location: {KEY_FILE}")
print("Keep the encryption key secure for future decryption.")
