import streamlit as st #Streamlit is used to quickly build web apps for data science and machine learning using only Python
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_ollama import OllamaEmbeddings                      
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain             
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()
groq_api_key = os.environ.get('Groq_document_rag_chatbox')
st.title("chatbox with llama3 Demo")

llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the following question based on the context provided.
    Think step by step before providing the answer.
    <context>
    {context}
    </context>"""),
    ("user", "Question: {input}")
])

user_input = st.text_input("Enter your question here") 

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding=OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = DirectoryLoader("./us_census",glob="**/*.pdf",loader_cls=PyPDFLoader) #Data ingestion from multiple pdf files in the folder
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)


if st.button("Document Embedding"):
    vector_embedding() #function called
    st.write("Vector Store is ready")

if user_input:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' button first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

