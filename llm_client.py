# llm_client.py – Weiterleitungsdatei. Logik liegt in llm_router.py.
# ollama_chat ist ein Alias für route_llm_call() – Name bleibt für bestehende Aufrufer.
from llm_router import ollama_is_up
from llm_router import route_llm_call as ollama_chat

__all__ = ["ollama_is_up", "ollama_chat"]
