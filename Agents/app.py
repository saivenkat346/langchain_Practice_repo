import os
import sys
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentType, initialize_agent, load_tools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import clean_response


# ========== Setup ==========
st.set_page_config(page_title="Best Finder Agent", page_icon="🔥", layout="centered")

# Load environment
load_dotenv()

# Choose local vs OpenAI cloud
USE_LOCAL = os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true"

if USE_LOCAL:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",  # Local LM Studio model
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")


# ========== Prompt Chain ==========
prompt = PromptTemplate(
    input_variables=["thing", "source"],
    template="""
You are an expert guide.  
Provide the **best {thing}** based on reliable knowledge from {source}.  
Explain why it is considered the best, and include practical insights.  
If there are multiple options, rank them and suggest the most effective one.  
    """
)

output_parser = StrOutputParser()
cleaner = RunnableLambda(clean_response)

# Build chain
chain = prompt | llm | output_parser | cleaner

# Tools + Agent
tools = load_tools(["wikipedia"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# ========== Streamlit Interface ==========
st.title("🔥 Best Finder Agent")
st.markdown("Ask me for the *best* of anything, and I'll fetch answers from reliable sources.")

# Input fields
thing = st.text_input("What do you want to find the **best** of?", placeholder="e.g. exercise to burn fat")
source = st.text_input("What source should I rely on?", placeholder="e.g. fitness science, Wikipedia, experts")

if st.button("Find Best"):
    if thing and source:
        with st.spinner("Thinking..."):
            query = chain.invoke({"thing": thing, "source": source})
            response = agent.run(query)
        st.subheader("✅ Best Recommendation")
        st.write(response)
    else:
        st.warning("⚠️ Please enter both *thing* and *source*.")
