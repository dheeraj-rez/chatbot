# main.py
from fastapi import FastAPI
from api.chatbot import router as chatbot_router

app = FastAPI(
    title="LangChain Loan Bot API (Production)",
    description="A structured API for a loan assistance chatbot powered by LangChain."
)

# Include the chatbot router with a prefix for versioning, e.g., /api/v1/chat
app.include_router(chatbot_router, prefix="/api/v1", tags=["Chatbot"])

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Loan Bot API!"}