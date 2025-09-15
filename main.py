from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Farming Assistant API")

# Backend URLs
BASE_URL = "http://localhost:8000"     # Optional custom backend
OLLAMA_URL = "http://localhost:11434"  # Ollama server


class ChatRequest(BaseModel):
    question: str
    model: str = "gemma3:1b"


def detect_language(text: str) -> str:
    """Detect if the input text is Malayalam or English."""
    malayalam_chars = 'അആഇഈഉഊഋഎഏഐഒഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറൻൺൽൾ'
    return 'malayalam' if any(char in text for char in malayalam_chars) else 'english'


def add_farming_context(prompt: str, language: str) -> str:
    """Add Kerala farming context and ensure replies are in the same language."""
    if language == 'malayalam':
        return (
            "നിങ്ങൾ ഒരു കാർഷിക വിദഗ്ധനാണ്. "
            "കേരളത്തിലെ കൃഷിക്കാരെ സഹായിക്കുക. "
            "മലയാളത്തിൽ മാത്രമേ ഉത്തരം നൽകരുത്, "
            "മറ്റെന്തെങ്കിലും ഭാഷയിൽ ഉത്തരം പറയരുത്.\n\n"
            + prompt
        )
    else:
        return (
            "You are a farming expert helping farmers in Kerala, India. "
            "Provide practical agricultural advice in English only.\n\n"
            + prompt
        )


@app.post("/chat")
def chat(request: ChatRequest):
    """Receive a farmer's question and return advice."""
    language = detect_language(request.question)
    enhanced_prompt = add_farming_context(request.question, language)

    # Try custom backend first
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"prompt": enhanced_prompt, "model": request.model},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "response": result.get("response", ""),
            "language": language
        }
    except requests.exceptions.RequestException:
        # Fallback to Ollama
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": request.model, "prompt": enhanced_prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            return {
                "success": True,
                "response": response.json().get("response", ""),
                "language": language
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e), "language": language}


@app.get("/")
def root():
    return {"message": "Farming Assistant API is running!"}                                 