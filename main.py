# main.py
from fastapi import FastAPI
from api.chatbot import router as chatbot_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LangChain Loan Bot API (Production)",
    description="A structured API for a loan assistance chatbot powered by LangChain."
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # The list of origins that are allowed to make cross-origin requests.
    allow_credentials=True, # Allows cookies to be included in requests.
    allow_methods=["*"],    # Allows all methods (GET, POST, etc.).
    allow_headers=["*"],    # Allows all headers.
)

# Include the chatbot router with a prefix for versioning, e.g., /api/v1/chat
app.include_router(chatbot_router, prefix="/api/v1", tags=["Chatbot"])

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Loan Bot API!"}