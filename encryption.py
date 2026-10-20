
import pandas as pd
from cryptography.fernet import Fernet
import os

# Step 1: Generate an encryption key if it does not exist
if not os.path.exists("secret.key"):
    key = Fernet.generate_key()

    with open("secret.key", "wb") as key_file:
        key_file.write(key)

    print("Encryption key created successfully!")

# Step 2: Load the existing encryption key
with open("secret.key", "rb") as key_file:
    key = key_file.read()

cipher = Fernet(key)

# Step 3: Read PII data from the CSV file
df = pd.read_csv("pii_data.csv", dtype=str)

print("\nOriginal PII Data:")
print(df)

# Step 4: Select the columns to encrypt
sensitive_columns = [
    "FullName",
    "Email",
    "Phone",
    "Address",
    "NIC"
]

# Step 5: Encrypt the sensitive data
for column in sensitive_columns:
    df[column] = df[column].apply(
        lambda value: cipher.encrypt(value.encode()).decode()
        if pd.notna(value) else value
    )

# Step 6: Save encrypted data to a new CSV file
df.to_csv("encrypted_pii_data.csv", index=False)

print("\nEncrypted PII Data:")
print(df)

print("\nEncryption completed successfully!")
print("Encrypted file: encrypted_pii_data.csv")
print("Keep secret.key secure for decryption.")
