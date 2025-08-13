# api/chatbot.py
import uuid
from fastapi import APIRouter, HTTPException
from models.chat import ChatRequest, ChatResponse
from services.langchain_service import langchain_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    """Handles chat requests by invoking the main LangChain service."""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Invoke the LangChain object. It automatically handles retrieval, prompting,
        # LLM calls, and memory. We pass the session_id in the config.
        response = langchain_service.conversational_rag_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response.get("answer", "I'm sorry, I encountered an issue and couldn't generate a response.")
        
    except Exception as e:
        print(f"Error invoking LangChain: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

    return ChatResponse(answer=answer, session_id=session_id)