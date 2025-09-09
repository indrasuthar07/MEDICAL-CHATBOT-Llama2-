from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Load data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "m-chat"

# Get embedding dimension
dimension = len(embeddings.embed_query("hello world"))

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

# Use LangChain’s PineconeVectorStore (new way)
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)
