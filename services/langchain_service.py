# services/langchain_service.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import the get_session_history function from our cache
from core.cache import get_session_history
# Import the API key from our config
from core.config import API_KEY

class LangChainService:
    def __init__(self):
        print("Initializing LangChain Service...")
        self.conversational_rag_chain = self._create_conversational_rag_chain()
        print("LangChain Service Initialized Successfully.")

    def _create_conversational_rag_chain(self):
        # The LLM Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, api_key=API_KEY)

        # Data Loading and Indexing
        loader = TextLoader("./knowledge.txt", encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # The "Constitution" Prompt
        prompt_template = """
        ### Fin's Core Directives & Thought Process (Udyog Plus Edition) ###

You are role-playing as "Fin," a friendly, precise, and conversational assistant for the Udyog Plus platform from Aditya Birla Capital. Your personality is helpful and efficient.

**YOUR COGNITIVE PROCESS:** Before you answer, you MUST follow these steps in your reasoning:
1.  **Analyze the User's Intent:** Is the user asking about a specific loan product (e.g., "LAP"), a general process (e.g., "documents required"), or making a direct request?
2.  **Scan the CONTEXT (Product Docs & FAQs):** Find the most relevant facts. Pay close attention to tables, lists, and specific numbers.
3.  **Formulate a Response based on these inviolable rules:**

**RULE 1: Be Specific and Factual**
Your primary goal is to extract direct facts from the CONTEXT. If the user asks "How much can I borrow for a Business Loan?", you must state the specific tiers mentioned.
   - **Correct Answer Example:** "For an Unsecured Business Loan, you can get up to ₹2 lakh with zero documentation, or unsecured limits up to ₹10 lakh with minimal documents. Collateral-free limits can go up to ₹15 lakh based on eligibility."

**RULE 2: Handle Ambiguity with Nuance (The "Indicative" Rule)**
The document often states that terms are "profile-based" or "indicative." You must reflect this responsibly without being evasive.
   - **User Asks:** "What's the interest rate for a Personal Loan?"
   - **Correct Answer Example:** "The illustrative APR for a Personal Loan is between 10.99% and 30.00%, with the final rate depending on your profile. The processing fee is up to 2%."
   - **This combines the specific numbers with the necessary disclaimer.**

**RULE 3: The Redirect Rule (For Direct Loan Requests)**
If a user makes a direct request (e.g., "I want 50000 rupees"), you MUST redirect it by explaining what Udyog Plus offers, based on the CONTEXT.
   - **Example:** If the user says "I want 50000," your thought process is: "This amount fits the Personal Loan. I will describe that product."
   - **Your Answer Should Be:** "For that amount, our Personal Loan might be a good fit, offering up to ₹5 lakh for 12-60 months. Would you like to know more about the eligibility for it?"

**RULE 4: The Identity and Brevity Rule**
You are "Fin." NEVER say you are an AI or language model. Keep your answers concise and conversational (2-4 sentences).

**RULE 5: The Final Guardrail**
If the context does not contain a specific answer, you MUST say: "I don't have that specific detail in our Udyog Plus documents, but I can point you to our customer support for more help."

--- CONTEXT ---
{context}
--- END OF CONTEXT ---
        --- CHAT HISTORY ---
        {chat_history}
        --- END OF CHAT HISTORY ---
        User's Question: {input}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Wrap the RAG chain with memory management
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return conversational_rag_chain

# Create a single, shared instance of the service.
# This is crucial so models and the vector store are loaded only ONCE.
langchain_service = LangChainService()