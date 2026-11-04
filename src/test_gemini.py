import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Find the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the API key
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the .env file.")

# Connect to Gemini
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Explain what an employment contract is in simple English."
    )

    print("\nGemini response:")
    print(response.text)

except Exception as error:
    print("\nGemini request failed.")
    print(error)
