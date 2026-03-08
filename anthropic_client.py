# anthropic_client.py – Weiterleitungsdatei. Logik liegt in llm_router.py.
from llm_router import anthropic_is_up, claude_chat, get_client

__all__ = ["anthropic_is_up", "claude_chat", "get_client"]
