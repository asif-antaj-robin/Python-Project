import streamlit as st
import os
import time

#Langchain iports
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


# Ensure directories
if not os.path.exists("pdfFiles"):
    os.makedirs("pdfFiles")
if not os.path.exists("vectorDB"):
    os.makedirs("vectorDB")

# Session Variables
if "template" not in st.session_state:
    st.session_state["template"] = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone must be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:"""

if "prompt" not in st.session_state:
    st.session_state["prompt"] = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state["template"],
    )

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = Chroma(
        persist_directory="vectorDB",
        embedding_function=OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="llama3:8b"
        )
    )

if "llm" not in st.session_state:
    st.session_state["llm"] = Ollama(
        base_url="http://localhost:11434",
        model="llama3:8b",
        verbose=True,
        callback_manager=CallbackManager(
            [StreamingStdOutCallbackHandler()]
        )
    )

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# UI Title
st.title("Chatbot - Talk to Your PDFs")

# File uploader
upload_file = st.file_uploader("Choose a PDF File", type="pdf")

# Show chat history
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Process Upload
if upload_file is not None:
    st.text("File uploaded successfully!")
    file_path = f"pdfFiles/{upload_file.name}"
    if not os.path.exists(file_path):
        with st.status("Saving file..."):
            bytes_data = upload_file.read()
            with open(file_path, "wb") as f:
                f.write(bytes_data)

            #Load and process the PDF
            loader = PyPDFLoader(file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len,
            )
            all_splits = text_splitter.split_documents(data)

            #Create Vectorstore
            st.session_state["vectorstore"] = Chroma.from_documents(
                documents=all_splits,
                embedding_function=OllamaEmbeddings(
                    base_url="http://localhost:11434",
                    model="llama3:8b"
                    )
            )
            st.session_state['vectorstore']

    # Set up retriever
    st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever()

    # QA Chain
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=st.session_state["llm"],
            chain_type="stuff",
            retriever=st.session_state["retriever"],
            verbose=True,
            chain_type_kwargs={
                "prompt": st.session_state["prompt"],
                "memory": st.session_state["memory"],
            },
        )

    # hendel user input and Chatbot interaction
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state["chat_history"].append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state["qa_chain"](user_input)
            message_placeholder = st.empty()
            full_resonse = ""
            for chunk in response['result'].split():
                full_resonse += chunk +" "
                time.sleep(0.05)
                message_placeholder.markdown(full_resonse + " ")
            message_placeholder.markdown(full_resonse)

        chatbot_message = {"role": "assistant", "message":response['result']}
        st.session_state["chat_history"].append(chatbot_message)

else:
    st.write("Please upload a PDF file to start chatbot.")