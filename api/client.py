import requests
import streamlit as st

st.title("Multi Model LangServe Chatbot")

# Model selector
model = st.selectbox("Choose Model", ["llama", "gemma"])

question = st.text_input("Ask your question")

def get_response(question, model):
    API_URL = f"http://localhost:8001/{model}/invoke"  
    response = requests.post(
        API_URL,
        json={"input": {"question": question}}
    )
    return response.json()["output"]

# UI trigger
if question:
    st.write(get_response(question, model))  