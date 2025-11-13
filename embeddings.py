
##This script generates embeddings for documents using OpenAI and LangChain.

```python
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Set your OpenAI and Pinecone API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))


#generate the embeddings by passing the documents
def generate_embeddings_from_documents(documents):
    # Load OpenAI embeddings from LangChain
    embeddings = OpenAIEmbeddings()

    # Create a Pinecone index if it doesn't exist
    index_name = "document-qa-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=embeddings.embedding_size)

    # Load documents into Pinecone index
    index = pinecone.Index(index_name)

    # Convert the documents into embeddings and insert into Pinecone
    embeddings_list = [embeddings.embed(document['text']) for document in documents]
    ids = [str(i) for i in range(len(documents))]

    index.upsert(vectors=zip(ids, embeddings_list))
    return index
