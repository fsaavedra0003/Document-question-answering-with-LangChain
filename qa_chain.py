from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate

def create_qa_chain(vector_store):
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question. Context: {context} Question: {question}"
    )

    # Create a LangChain retrieval-based QA system using the vector store (Pinecone)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def get_answer(qa_chain, query):
    # Get the answer from the QA chain
    return qa_chain.run(query)
