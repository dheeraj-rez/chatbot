# core/cache.py
from typing import Dict
from langchain.memory import ChatMessageHistory

# This simple dictionary will hold all active LangChain chat history objects.
# Key: session_id (str), Value: A ChatMessageHistory object.
chat_history_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates a chat history for a given session ID."""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]