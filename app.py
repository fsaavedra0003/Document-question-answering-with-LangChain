import streamlit as st
from embeddings import generate_embeddings_from_documents
from vector_store import create_vector_store, query_vector_store
from qa_chain import create_qa_chain, get_answer
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pinecone and LangChain
vector_store = create_vector_store()
qa_chain = create_qa_chain(vector_store)

# Streamlit UI
st.title("Document Q&A with LangChain + Pinecone + Streamlit")
st.write("Upload your document and ask any question!")

uploaded_file = st.file_uploader("Choose a document", type="txt")
if uploaded_file is not None:
    # Read the uploaded document
    document_text = uploaded_file.read().decode("utf-8")
    documents = [{"text": document_text}]
    st.text_area("Document Text", document_text, height=200)

    # Generate embeddings and store them in Pinecone
    index = generate_embeddings_from_documents(documents)
    st.success("Document indexed successfully!")

# Get user query
query = st.text_input("Ask a question")
if query:
    # Get the answer from the QA chain
    answer = get_answer(qa_chain, query)
    st.write("Answer: ", answer)
