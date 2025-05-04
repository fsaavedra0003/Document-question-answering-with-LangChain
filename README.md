# 📄 Document Q&A with LangChain + Pinecone + Streamlit


This project allows users to upload documents and query them for specific answers using  the framework of  LangChain, Pinecone vector database, and Streamlit web framework . The system performs the following:

- Upload documents to be indexed.
- Convert documents into embeddings using LangChain.
- Store embeddings in Pinecone for efficient querying.
- Use Streamlit for an interactive interface to ask questions and get answers.

### 🔍 Features

- Document upload and indexing.
- Text embeddings generation using LangChain.
- Search for answers using Pinecone similarity search.
- Real-time interactive Q&A with a Streamlit frontend.

## Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/fsaavedra0003/Document-QA-LangChain-Pinecone-Streamlit.git
cd Document-QA-LangChain-Pinecone-Streamlit
pip install -r requirements.txt
