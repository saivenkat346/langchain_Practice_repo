from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import clean_response

import streamlit as st
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# langsmit tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Toggle between OpenAI cloud vs LM Studio
USE_LOCAL = os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true"

if USE_LOCAL:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",  # replace with your local model name
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")


# prmpt Templete
messages = [
    (
        "system",
        "you are  helpful assistant. please response to the user queries in english",
    ),("user", "Question:{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

# streamlit framework

st.title("Langchain Demo With OpenAI API")
input_text = st.text_input("Search the topic you want")

# openAI llm
output_parser = StrOutputParser()
cleaner = RunnableLambda(clean_response)
chain = prompt | llm | output_parser | cleaner

if input_text: 
    st.write(chain.invoke({"question": input_text}))
