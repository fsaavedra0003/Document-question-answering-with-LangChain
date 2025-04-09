# main.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import pinecone
import os

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "doc-qa-index"

# Create vectorstore
embedding = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)

# Setup QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Streamlit UI
st.title("📄 Document Q&A with LangChain + Pinecone")
question = st.text_input("Ask a question about the documents")

if question:
    with st.spinner("Searching for answer..."):
        result = qa.run(question)
        st.success(result)
