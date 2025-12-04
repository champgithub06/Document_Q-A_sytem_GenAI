import os
import os
import gradio as gr
from dotenv import load_dotenv

# Libraries
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- CONFIGURATION ---
CHROMA_PATH = r"chroma_db"

os.environ["GOOGLE_API_KEY"] = "AIzaSyA1z-VN6Grq3d0QcsuXFNrR-rmGF8gt5-U"

# 1. Setup Embeddings 
print("Loading Embeddings...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Setup LLM (Free - Google Cloud)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.5,
    max_retries=2,
)

# 3. Connect to Database
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# 4. Setup Retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

def stream_response(message, history):
    print(f"Searching for: {message}")
    
    try:
        # Retrieve docs
        docs = retriever.invoke(message)
        knowledge = ""
        for doc in docs:
            knowledge += doc.page_content + "\n\n"

        # Generate Answer
        if message:
            partial_message = ""
            
            rag_prompt = f"""
            You are a helpful assistant. 
            First, check the "Context" below to answer the question. 
            If the answer is in the context, use it to answer.
            If the answer is NOT in the context, use your own general knowledge to answer kindly.
            Question: {message}
            
            Context:
            {knowledge}
            """

            # Stream the response from Gemini
            for chunk in llm.stream(rag_prompt):
                partial_message += chunk.content
                yield partial_message
                
    except Exception as e:
        yield f"Error: {str(e)}"

# Initiate Gradio
chatbot = gr.ChatInterface(
    stream_response, 
    textbox=gr.Textbox(placeholder="Ask your PDF...", container=False, autoscroll=True, scale=7),
    title="PDF Chatbot (Powered by Gemini 1.5 Flash)"
)
