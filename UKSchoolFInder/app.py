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

# Load .env variables
load_dotenv()

# Toggle between OpenAI cloud vs LM Studio
USE_LOCAL = os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true"

if USE_LOCAL:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",  # Local LM Studio model
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

# Template for finding best schools
UK_Best_school_prompt_template = PromptTemplate(
    input_variables=["location", "school_type", "requirements"],
    template="""
You are an expert UK education consultant. A user is looking for the best school.

Location: {location}  
School Type: {school_type} (examples: state, private, boarding, international)  
Special Requirements: {requirements}

Task:  
1. List the top recommended schools in the given location.  
2. Briefly explain why each school is a good fit (academics, facilities, reputation, Ofsted rating, etc.).  
3. Mention admission requirements (documents, visa if applicable, age criteria).  
4. Provide links or references if possible.  

Answer clearly, in a structured format with bullet points.
"""
)

output_parser = StrOutputParser()
cleaner = RunnableLambda(clean_response)

findingBestSchoolChain = UK_Best_school_prompt_template | llm | output_parser | cleaner

# Streamlit UI
st.title("Finding Best School in UK")
location = st.text_input("Enter your location")
school_type = st.text_input("Enter School Type")
requirements = st.text_area("Mention requirements (e.g., student inside/outside UK, health needs, etc.)")

if st.button("Find Schools"):
    if location and school_type and requirements:
        response = findingBestSchoolChain.invoke({
            "location": location,
            "school_type": school_type,
            "requirements": requirements
        })
        st.markdown(response)
    else:
        st.warning("Please fill all fields before searching.")
