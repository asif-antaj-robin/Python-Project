from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama


load_dotenv()

# #environment variables call
# os.environ["OPENAI_API_KEY"]  =os.getenv("LANGCHAIN_API_KEY")

#langsmith.tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#create_chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide responses to the user queries."),
    ("user", "question: {question}")
])

#openai
# llm = ChatOpenAI(model = "gpt-3.5-turbo")
llm = Ollama(model="gemma:2b")
OutputParser = StrOutputParser()

#chain
Chain = prompt|llm|OutputParser

#streamlit frames
st.title("Assistant ChatBot")
input_text = st.text_input("searcs the topic you want")

if input_text:
    st.write(Chain.invoke({"question" : input_text}))


