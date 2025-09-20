# LangChain Practice Project

A comprehensive collection of LangChain applications demonstrating various AI/ML concepts including chatbots, agents, RAG workflows, and API services.

## 🚀 Project Overview

This project contains multiple independent applications showcasing different LangChain capabilities:

- **Basic Chatbot** - Simple conversational AI
- **Chains Practice** - Sequential processing chains
- **Best Finder Agent** - AI agent with Wikipedia tools
- **RAG Workflow** - Document question-answering system
- **UK School Finder** - Specialized prompt engineering
- **API Server + Client** - Microservices with LangServe

## 📁 Project Structure

```
langchainPractice/
├── requirements.txt              # Python dependencies
├── helper.py                    # Utility functions
├── basicChatbot/
│   └── app.py                   # Simple chatbot application
├── chainsPractice/
│   └── app.py                   # Sequential chains demonstration
├── Agents/
│   └── app.py                   # Best Finder Agent with Wikipedia
├── RAGWorkflow/
│   ├── app.py                   # Document Q&A system
│   └── dataSources/
│       ├── pdf/                 # PDF documents (empty)
│       └── text/
│           └── speech.txt       # MLK "I've Been to the Mountaintop" speech
├── UKSchoolFInder/
│   └── app.py                   # UK school finder application
└── api/
    ├── app.py                   # FastAPI server
    └── client.py                # Streamlit client for API
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Git (optional, for cloning)

### Step-by-Step Installation

1. **Install Python**
   ```bash
   # Download the latest Python from the official website
   # https://www.python.org/downloads/
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate Virtual Environment**
   
   **Windows (PowerShell):**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   
   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Environment Configuration**
   
   Create a `.env` file in the project root:
   ```env
   # OpenAI Configuration (if using OpenAI API)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # LangSmith Tracking (optional)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   
   # Local LM Studio Configuration
   USE_LOCAL_LMSTUDIO=true
   ```

6. **LM Studio Setup (if using local mode)**
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load a compatible model (e.g., `deepseek/deepseek-r1-0528-qwen3-8b`)
   - Start the local server on `http://localhost:1234`

## 🚀 Running Applications

### 1. Basic Chatbot
```bash
cd basicChatbot
streamlit run app.py
```
**Access:** http://localhost:8501

### 2. Chains Practice
```bash
cd chainsPractice
streamlit run app.py
```
**Access:** http://localhost:8501

### 3. Best Finder Agent
```bash
cd Agents
streamlit run app.py
```
**Access:** http://localhost:8501

### 4. RAG Workflow (Document Q&A)
```bash
cd RAGWorkflow
streamlit run app.py
```
**Access:** http://localhost:8501

### 5. UK School Finder
```bash
cd UKSchoolFInder
streamlit run app.py
```
**Access:** http://localhost:8501

### 6. API Server + Client
**Terminal 1 - Start API Server:**
```bash
cd api
python app.py
```
**Access:** http://localhost:8000

**Terminal 2 - Start Client:**
```bash
cd api
streamlit run client.py
```
**Access:** http://localhost:8501

## 🔧 Configuration Options

- **Local Mode (Default):** Uses LM Studio with model `deepseek/deepseek-r1-0528-qwen3-8b`
- **OpenAI Mode:** Set `USE_LOCAL_LMSTUDIO=false` in `.env` file
- **API Keys:** Configure in `.env` file for OpenAI and LangSmith tracking

## 📚 Application Details

### Basic Chatbot
- Simple conversational AI using LangChain
- Supports both OpenAI API and local LM Studio
- Clean response processing

### Chains Practice
- Demonstrates sequential processing chains
- Two modes: Simple Chain and Sequential Chain
- Title generation → Paragraph expansion workflow

### Best Finder Agent
- AI agent with Wikipedia search capabilities
- Finds "best" recommendations for any topic
- Uses LangChain agents and tools

### RAG Workflow
- Document upload and processing (PDF/TXT)
- Vector search and retrieval
- Question-answering over documents
- Configurable chunk size and overlap

### UK School Finder
- Specialized prompt engineering
- Location, school type, and requirements input
- Structured recommendations with admission details

### API Server + Client
- FastAPI backend with LangServe
- Essay and poem generation endpoints
- Streamlit frontend client
- Microservices architecture

## 🛠️ Dependencies

Key dependencies include:
- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools and loaders
- `streamlit` - Web application framework
- `fastapi` - API framework
- `chroma` - Vector database
- `sentence-transformers` - Embeddings

## 📝 Usage Examples

### RAG Workflow
1. Upload PDF or TXT documents
2. Adjust chunk size and overlap settings
3. Process documents to create vector store
4. Ask questions about the content
5. View answers with source citations

### Best Finder Agent
1. Enter what you want to find the "best" of
2. Specify your preferred source (e.g., "fitness science")
3. Get AI-powered recommendations with explanations

### API Server
1. Start the FastAPI server
2. Use the Streamlit client or make direct API calls
3. Generate essays and poems on any topic

## 🤝 Contributing

Feel free to explore and modify the applications to learn more about LangChain concepts!

## 📄 License

This project is for educational purposes and LangChain practice.
