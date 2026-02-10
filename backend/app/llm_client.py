import os
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()

def get_client() -> Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")

    return Client(api_key=api_key)

def call_llm(prompt: str) -> str | None:
    try:
        client = get_client()  # lazy initialization (IMPORTANT)

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        print(f"[LLM ERROR]: {e}")
        return None