from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import uvicorn

app = FastAPI(
    title="Multi Model Server",
    version="1.0"
)

# ---------------- Llama Chain ----------------
llama_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Provide detailed answers to the user questions."),
        ("user", "Question: {question}")
    ]
)

llama_llm = Ollama(model="llama3.2")
llama_chain = llama_prompt | llama_llm | StrOutputParser()

add_routes(
    app,
    llama_chain,
    path="/llama"
)

# ---------------- Gemma Chain ----------------
gemma_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a short and funny assistant. Give answers in funny way."),
        ("user", "Question: {question}")
    ]
)

gemma_llm = Ollama(model="gemma3")
gemma_chain = gemma_prompt | gemma_llm | StrOutputParser()

add_routes(
    app,
    gemma_chain,
    path="/gemma"
)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)

