from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn

import os
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import clean_response

# Load variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Toggle between OpenAI cloud vs LM Studio
USE_LOCAL = os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true"

if USE_LOCAL:
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
    )
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

# Disable OpenAPI docs generation to avoid schema issues with LM Studio
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server",
    docs_url=None if USE_LOCAL else "/docs",  # Disable docs for local mode
    redoc_url=None if USE_LOCAL else "/redoc",  # Disable redoc for local mode
    openapi_url=None if USE_LOCAL else "/openapi.json"  # Disable openapi for local mode
)

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")
output_parser = StrOutputParser()

chain1 = prompt1 | llm | output_parser
chain2 = prompt2 | llm | output_parser

add_routes(
    app,
    chain1,
    path="/essay"
)
add_routes(
    app,
    chain2,
    path="/poem"
)

@app.get("/")
def read_root():
    routes = ["/essay", "/poem"]
    if not USE_LOCAL:
        routes.append("/docs")
    return {
        "message": "Langchain Server is running!", 
        "available_routes": routes,
        "playgrounds": ["/essay/playground/", "/poem/playground/"],
        "mode": "Local LM Studio" if USE_LOCAL else "OpenAI Cloud"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)