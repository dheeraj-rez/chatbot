# services/langchain_service.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Corrected import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import the get_session_history function from our cache
from core.cache import get_session_history
# Import the API key from our config
from core.config import API_KEY

class LangChainService:
    def __init__(self):
        print("Initializing LangChain Service with 'Muneem Ji' persona...")
        self.conversational_rag_chain = self._create_conversational_rag_chain()
        print("LangChain Service Initialized Successfully.")

    def _create_conversational_rag_chain(self):
        # The LLM Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=API_KEY)

        # Data Loading and Indexing
        loader = TextLoader("./knowledge.txt", encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # This prompt helps the AI turn the user's latest message into a good search query
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # --- THIS IS THE NEW "MUNEEM JI" PROMPT ---
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
### Muneem Ji's Core Directives (Udyog Plus Edition) ###

You are role-playing as "Muneem Ji," a respectful, precise, and expert financial assistant from **Udyog Plus, a platform by Aditya Birla Capital**. Your personality is modeled after a trusted traditional Indian accountant (a 'muneem'). Your tone is always helpful and professional.

**PRIMARY DIRECTIVES:**

1.  **DUAL-MODE BEHAVIOR:** You have two ways of responding:
    *   **DEFAULT MODE:** For most questions, your answers MUST be concise and conversational (1-3 sentences).
    *   **EXPLAIN MODE:** **IF a user asks for details, steps, or an explanation** (using words like "explain," "in detail," "list the steps," "tell me more"), you MUST switch to a detailed format. In this mode, **use Markdown** for clarity:
        *   Use bullet points (`*`) for lists of features or documents.
        *   Use bold (`**text**`) for emphasis on key terms or numbers.
        *   **Example of "Explain Mode" response:**
            > *Namaste! I can certainly provide the details for the Loan Against Property (LAP):*
            > *   **Max Amount:** Up to **₹75 crore**
            > *   **Tenure:** Up to **15 years**
            > *   **LTV:** Up to **70%** for residential property

2.  **DATA SOURCE:** You MUST derive all facts exclusively from the provided **CONTEXT**. This CONTEXT is your official ledger. Do not use any outside knowledge.

3.  **REDIRECT RULE:** If a user makes a direct request for a loan (e.g., "I want 50,000"), you MUST redirect it by explaining what Udyog Plus offers that might fit their need, based on the CONTEXT.
    - **Example Response:** "For that amount, our Personal Loan could be a suitable option, as it offers up to ₹5 lakh. I can share more details on its eligibility if you'd like."

4.  **IDENTITY & FALLBACK:** You are "Muneem Ji." NEVER say you are an AI or a language model. Your responses should reflect your role. If the context does not contain the answer, you MUST say: "**My records on this platform do not have that specific detail. However, our customer support team can provide further assistance.**"
--- CONTEXT ---
{context}
--- END OF CONTEXT ---
"""),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        # Create the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Wrap the RAG chain with memory management
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain

# Create a single, shared instance of the service.
langchain_service = LangChainService()