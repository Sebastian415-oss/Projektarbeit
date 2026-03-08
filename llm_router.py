# llm_router.py
# Einzige Stelle wo LLM-Aufrufe stattfinden. Claude (Anthropic) hat Vorrang –
# wenn ANTHROPIC_API_KEY gesetzt ist, wird Ollama komplett ignoriert.
# route_llm_call() ist die öffentliche API; llm_client.py re-exportiert sie als ollama_chat()
# damit bestehende Aufrufer nichts ändern müssen.

import os
import time
import requests
import anthropic
from typing import Any, Dict, List, Optional

# -------------------------
# Anthropic (Claude)
# -------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

_client = None


def anthropic_is_up() -> bool:
    return bool(ANTHROPIC_API_KEY)


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def claude_chat(
    messages: List[Dict[str, str]],
    system: str = "",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    client = get_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    content = response.content[0].text if response.content else ""
    return {
        "message": {
            "content": content
        },
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# -------------------------
# Ollama
# -------------------------

OLLAMA_URL = "http://localhost:11434"


def ollama_is_up(timeout_s: int = 2) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


# -------------------------
# Router
# -------------------------

def route_llm_call(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    timeout_s: int = 120,
    num_predict: int = 512,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Routet LLM-Aufrufe: Claude (Anthropic) hat Vorrang, Ollama ist Fallback.
    """
    if anthropic_is_up():
        try:
            sys_msg = next((m["content"] for m in messages
                            if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            return claude_chat(
                messages=user_msgs,
                system=sys_msg,
                max_tokens=num_predict,
                temperature=temperature,
            )
        except Exception as e:
            print(f"Anthropic Fehler, Fallback Ollama: {e}")

    # Ollama unterstützt keine system-role im messages-Array – wird herausgefiltert
    ollama_messages = [m for m in messages if m["role"] != "system"]

    url = f"{OLLAMA_URL}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))  # linearer Backoff: 1.5s, 3.0s, ...
                continue
            raise RuntimeError(f"LLM call failed after retries: {e}") from e

    raise RuntimeError(f"LLM call failed: {last_err}")
