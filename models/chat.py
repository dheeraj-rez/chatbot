# models/chat.py
from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(
        None, description="Unique ID for the conversation. If none, a new session is created."
    )

class ChatResponse(BaseModel):
    answer: str
    session_id: str