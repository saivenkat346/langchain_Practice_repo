import os
import sys
import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load .env variables
load_dotenv()

# Add helper path if it exists
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from helper import clean_response
except ImportError:
    # Create a simple clean_response function if helper is not available
    def clean_response(response):
        return response

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Document Question & Answer System")
st.markdown("Upload your documents (TXT or PDF) and ask questions about their content!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Toggle between OpenAI cloud vs LM Studio
    use_local = st.toggle(
        "Use Local LM Studio", 
        value=os.getenv("USE_LOCAL_LMSTUDIO", "true").lower() == "true",
        help="Toggle between OpenAI API and local LM Studio"
    )
    
    st.markdown("---")
    st.header("📊 Document Processing Settings")
    
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def initialize_llm_and_embeddings(use_local):
    """Initialize LLM and embeddings based on configuration"""
    if use_local:
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-0528-qwen3-8b",  # Local LM Studio model
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings()
    
    return llm, embeddings

def load_documents(uploaded_files, chunk_size, chunk_overlap):
    """Load and process uploaded documents"""
    documents = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load based on file type
            if uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            elif uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue
            
            # Load the document
            docs = loader.load()
            
            # Add source information to each document
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            
            documents.extend(docs)
            
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs
    
    return []

def create_vectorstore_and_chain(documents, llm, embeddings):
    """Create vector store and retrieval chain"""
    # Create vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""
        Based on the following context from the document(s), please answer the user's question:
        
        Context: {context}
        
        Question: {input}
        
        Answer:
        """
    )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return vectorstore, retrieval_chain

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload one or more TXT or PDF files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
    
    # Process documents button
    if st.button("🔄 Process Documents", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            try:
                # Initialize LLM and embeddings
                llm, embeddings = initialize_llm_and_embeddings(use_local)
                
                # Load and process documents
                documents = load_documents(uploaded_files, chunk_size, chunk_overlap)
                
                if documents:
                    # Create vectorstore and retrieval chain
                    vectorstore, retrieval_chain = create_vectorstore_and_chain(
                        documents, llm, embeddings
                    )
                    
                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retrieval_chain = retrieval_chain
                    st.session_state.documents_processed = True
                    
                    st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                else:
                    st.error("❌ No documents could be processed")
                    
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

with col2:
    st.header("💬 Ask Questions")
    
    if st.session_state.documents_processed:
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="Ask anything about your uploaded documents...",
            height=100
        )
        
        # Query button
        if st.button("🔍 Get Answer", disabled=not question.strip()):
            if st.session_state.retrieval_chain:
                with st.spinner("Generating answer..."):
                    try:
                        # Get response from retrieval chain
                        response = st.session_state.retrieval_chain.invoke({"input": question})
                        
                        # Display answer
                        st.subheader("🤖 Answer:")
                        answer = clean_response(response.get('answer', 'No answer found'))
                        st.write(answer)
                        
                        # Display source documents
                        if 'context' in response:
                            with st.expander("📖 Source Documents"):
                                for i, doc in enumerate(response['context']):
                                    st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is the main topic of these documents?",
            "Can you provide a summary?",
            "What are the key points mentioned?",
            "Are there any specific dates or numbers mentioned?"
        ]
        
        for i, sample in enumerate(sample_questions):
            if st.button(f"📝 {sample}", key=f"sample_{i}"):
                if st.session_state.retrieval_chain:
                    with st.spinner("Generating answer..."):
                        try:
                            response = st.session_state.retrieval_chain.invoke({"input": sample})
                            st.subheader("🤖 Answer:")
                            answer = clean_response(response.get('answer', 'No answer found'))
                            st.write(answer)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Please upload and process documents first to start asking questions!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with ❤️ using Streamlit, LangChain, and OpenAI/LM Studio
    </div>
    """,
    unsafe_allow_html=True
)