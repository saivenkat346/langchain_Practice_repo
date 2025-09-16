import requests
import streamlit as st
import sys
import os

# Add parent dir to sys.path to import helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import clean_response


# ---- API CALLER ----
def call_langserve(endpoint: str, topic: str):
    """Generic function to call LangServe endpoints."""
    try:
        response = requests.post(
            f"http://localhost:8000/{endpoint}/invoke",
            json={"input": {"topic": topic}}
        )
        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}"

        data = response.json()
        return data.get("output", data)  # Prefer 'output' key
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ---- STREAMLIT APP ----
st.set_page_config(page_title="LangChain Demo", page_icon="📚")
st.title("📚 LangChain Demo with LM Studio DeepSeek")

st.write("Generate AI-written essays and poems using local LangServe endpoints.")

# Tabs for better organization
tab1, tab2 = st.tabs(["✍️ Essay Generator", "🎭 Poem Generator"])

with tab1:
    topic = st.text_input("Enter essay topic:")
    if st.button("Generate Essay", use_container_width=True):
        if topic.strip():
            with st.spinner("Generating essay..."):
                result = call_langserve("essay", topic)
                st.subheader("Essay Result")
                st.write(clean_response(result))
        else:
            st.warning("⚠️ Please enter a topic first.")

with tab2:
    topic = st.text_input("Enter poem topic:")
    if st.button("Generate Poem", use_container_width=True):
        if topic.strip():
            with st.spinner("Generating poem..."):
                result = call_langserve("poem", topic)
                st.subheader("Poem Result")
                st.write(clean_response(result))
        else:
            st.warning("⚠️ Please enter a topic first.")
