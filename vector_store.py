import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pinecone with your API key and environment
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

def create_vector_store():
    index_name = "document-qa-index"
    # Create a vector store in Pinecone
    return Pinecone(index_name=index_name)

def query_vector_store(query_text, vector_store):
    # Query the vector store (Pinecone) for the most relevant document embeddings
    result = vector_store.similarity_search(query_text, k=1)  # k = number of results to return
    return result
