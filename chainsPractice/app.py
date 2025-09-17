import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import clean_response
# Load environment
load_dotenv()

# LLM Setup
USE_LOCAL = os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true"

if USE_LOCAL:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

output_parser = StrOutputParser()
cleaner = RunnableLambda(clean_response)

# Prompts
title_prompt = PromptTemplate.from_template("Write a title about {topic}")
paragraph_prompt = PromptTemplate.from_template("Expand this title into a paragraph: {title}")

# Simple Sequential Chain - only returns final output
def simple_chain(topic):
    title = (title_prompt | llm | output_parser|cleaner).invoke({"topic": topic})
    paragraph = (paragraph_prompt | llm | output_parser|cleaner).invoke({"title": title})
    return paragraph

# Sequential Chain - returns both title and paragraph  
def sequential_chain(topic):
    title = (title_prompt | llm | output_parser|cleaner).invoke({"topic": topic})
    paragraph = (paragraph_prompt | llm | output_parser|cleaner).invoke({"title": title})
    return {"title": title, "paragraph": paragraph}

# Streamlit Interface
st.title("LangChain Sequential Chains")

chain_type = st.selectbox("Choose Chain Type:", ["Simple Chain", "Sequential Chain"])
topic = st.text_input("Enter topic:")

if st.button("Generate") and topic:
    with st.spinner("Generating..."):
        try:
            if chain_type == "Simple Chain":
                result = simple_chain(topic)
                st.write("**Result:**")
                st.write(result)
            else:
                result = sequential_chain(topic)
                st.write("**Title:**")
                st.info(result["title"])
                st.write("**Paragraph:**")
                st.write(result["paragraph"])
        except Exception as e:
            st.error(f"Error: {e}")