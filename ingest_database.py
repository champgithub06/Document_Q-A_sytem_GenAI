

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# --- GOOGLE GEMINI EMBEDDINGS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="all-MiniLM-L6-v2")

# Create vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load PDFs
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(raw_documents)

# Create IDs
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add to vector DB
vector_store.add_documents(documents=chunks, ids=uuids)

print("Ingestion complete!")
