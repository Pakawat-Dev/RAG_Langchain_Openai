from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

# 1. Load documents
pdf_folder_path = r"D:\Safety_Docs"
try:
    loader = DirectoryLoader(pdf_folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} documents")
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []

#print(documents)

# 2. Split documents into chunks
text_splitter =RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
chunks = text_splitter.split_documents(documents)

#for i, chunk in enumerate(chunks):
    #print(f"chunk : {i+1}, {chunk.page_content}")

# 3. Convert data to vectors
embeddings = OpenAIEmbeddings()
#data = embeddings.embed_documents(["Zensorium"])
#print(data)

# 4. Store data in vector store
vectorstore =FAISS.from_documents(chunks, embeddings)

# 5. Retrieve data from vector store
retrievers = vectorstore.as_retriever()

prompt =ChatPromptTemplate.from_messages([
    ("system", "You are a safety expert with deep knowledge of Nitto Group's safety policies, procedures, and best practices. "
               "Your role is to provide safety-related information and guidance to employees. "
               "Ensure they understand and adhere to Nitto's safety policies. "
               "Use the provided context from Nitto safety documents to answer questions accurately and professionally. "
               "Always prioritize employee safety and well-being in your responses."),
    ("human", "Question: {question}"),
    ("ai", "Relevant Context: {context}"),
]) 

# Model
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Chain
rag_chain = (
    {"context": retrievers, "question": RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)

# Test the chain with error handling
try:
    result = rag_chain.invoke("Summarize for core Safety Principle?")
    print("RAG chain test successful")
except Exception as e:
    print(f"Error testing RAG chain: {e}")
    result = "Error occurred during initialization"

#print(result)

# Streamlit interface
st.title("Safety Chatbot")
st.write("Ask questions about Nitto Safety Practices based on the provided reference document.")

question = st.text_area("Provide your question here...", height=100)

if st.button("Ask anything about Safety!"):
    if question:
        try:
            result = rag_chain.invoke(question)
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")