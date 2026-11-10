import requests

# Ollama local API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# Model name
MODEL_NAME = "qwen3:1.7b"


def ask_local_llm(question):

    prompt = f"""
You are an assistant helping analyse Norwegian employment contracts.

Answer the following question using simple English.
Do not invent legal requirements.

Question: {question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 256,
                "num_ctx": 2048,
                "temperature": 0.1
            }
        },
        timeout=600
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


if __name__ == "__main__":

    question = input("Enter your question: ").strip()

    if question:
        answer = ask_local_llm(question)

        print("\n========== QWEN RESPONSE ==========\n")
        print(answer)

    else:
        print("Please enter a question.")