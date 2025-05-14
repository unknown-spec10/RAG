"""Agentic RAG - Streamlit Frontend"""
import streamlit as st
import os
import sys
from typing import List, Dict, Any
import tempfile
import json
import time
import uuid
import numpy as np  # For array operations with embeddings

# Detect if we're running on Streamlit Cloud
# This helps with proper resource configuration
if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_RUN_ON_SAVE"):
    os.environ["IS_STREAMLIT_CLOUD"] = "true"

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Import project modules after path setup
import src.pdf_processor.pdf_parser as pdf_parser
import src.pdf_processor.text_chunker as text_chunker
import src.embeddings.embedding_model as embedding_model
import src.vector_db.faiss_db as faiss_db
import src.rag.retriever as retriever
import src.rag.context_protocol as context_protocol
import src.agents.rag_agent as rag_agent
import src.utils.cache as cache_utils
from src.rag.agentic_workflow import ingest_document, retrieve_context

# Define configuration directly
APP_CONFIG = {
    "pdf_processor": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "embeddings": {
        # Use a small, efficient model to reduce memory usage
        # all-MiniLM-L6-v2 is only ~80MB and works well for general purpose embedding
        "model_name": "all-MiniLM-L6-v2",
    },
    "vector_db": {
        "storage_type": "local",
        "persist_directory": "data/faiss_db",
        "collection_name": "documents",
        "dimension": 384,
    },
    "retriever": {
        "top_k": 5,
        "rerank": True,
    },
    "llm": {
        # Use LLama3 models which are currently supported by Groq
        "model_name": "llama3-70b-8192",
        "provider": "groq",
    }
}

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pdf_files = []
    st.session_state.indexed_files = set()
    st.session_state.messages = []
    st.session_state.followup_questions = []
    # Create data directories
    os.makedirs("data/cache", exist_ok=True)
    # Initialize configuration
    st.session_state.config = APP_CONFIG

def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables."""
    # Default API keys - hardcoded for development only 
    # In production these should be set via environment variables or secrets
    groq_api_key = "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"
    
    # First try to load from .env file if running locally
    if not os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit_sharing":
        from dotenv import load_dotenv
        load_dotenv()
    
    # Check for keys in environment variables and Streamlit secrets
    # For deployed apps (Streamlit Cloud)
    if os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit_sharing":
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, TypeError):
            pass  # Use hardcoded keys as fallback
    else:
        # For local development, try environment variables
        if os.environ.get("GROQ_API_KEY"):
            groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Display info about LLM provider
    st.sidebar.info("✅ Using Groq API for LLM and hash-based embeddings for vector search")
    
    # We don't need google_api_key anymore since we're using hash-based embeddings
    return groq_api_key, ""  # Return empty string for google_api_key

def initialize_components():
    """Initialize RAG components."""
    try:
        # Get API keys first
        groq_api_key, google_api_key = get_api_keys()
        
        # Initialize file cache
        st.session_state.file_cache = cache_utils.FileCache("data/cache")
        
        # Initialize components in order
        st.session_state.pdf_parser = pdf_parser.PDFParser()
        st.session_state.text_chunker = text_chunker.TextChunker(
            chunk_size=st.session_state.config['pdf_processor']['chunk_size'],
            chunk_overlap=st.session_state.config['pdf_processor']['chunk_overlap']
        )
        
        # Initialize embedding model (using hash-based embeddings)
        st.session_state.embedding_model = embedding_model.EmbeddingModel()

        # Initialize vector database
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        for folder in ["pdfs", "faiss_db", "cache"]:
            os.makedirs(os.path.join(data_dir, folder), exist_ok=True)
            
        faiss_persist_dir = os.path.join(data_dir, "faiss_db")
        st.session_state.vector_db = faiss_db.FAISSVectorDB(
            persist_directory=faiss_persist_dir,
            collection_name="documents",
            dimension=st.session_state.config['vector_db']['dimension']
        )

        # Initialize remaining components
        st.session_state.context_protocol = context_protocol.ContextProtocolManager()
        st.session_state.retriever = retriever.RAGRetriever(
            vector_db=st.session_state.vector_db,
            embedding_model=st.session_state.embedding_model,
            top_k=st.session_state.config['retriever']['top_k']
        )

        # Use Groq model as default
        llm_name = "mixtral-8x7b"
        
        # Initialize RAG agent
        st.session_state.agent = rag_agent.RAGAgent(
            retriever=st.session_state.retriever,
            model_name=llm_name,
            context_protocol=st.session_state.context_protocol
        )
        
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.session_state.initialized = False
        return False

def render_chat():
    """Render the chat interface."""
    st.header("Chat with your Knowledge Base")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(source["content"])
                        if "metadata" in source and source["metadata"]:
                            st.markdown(f"*Metadata: {json.dumps(source['metadata'], indent=2)}*")
    
    # Display follow-up suggestions
    if st.session_state.followup_questions:
        st.markdown("**Suggested follow-up questions:**")
        cols = st.columns(min(3, len(st.session_state.followup_questions)))
        for i, question in enumerate(st.session_state.followup_questions[:3]):
            if cols[i].button(question, key=f"followup_{i}"):
                process_query(question)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        process_query(prompt)

def process_query(query: str):
    """Process a user query using agentic workflow."""
    if not st.session_state.initialized:
        st.error("Please upload and index some documents first.")
        return
    user_id = st.text_input("Enter your User ID for personalized answers:", key="query_user_id")
    if not user_id:
        st.warning("Please enter your User ID to query your policy and general info.")
        return
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        try:
            with st.spinner("Searching knowledge base..."):
                context_chunks = retrieve_context(query, user_id)
                # Compose context for LLM
                context_text = "\n\n".join([chunk["text"] for chunk in context_chunks if "text" in chunk])
                # Call LLM (simulate via st.session_state.agent for now)
                result = st.session_state.agent.query(query, context=context_text)
            response = result["response"]
            sources = result.get("sources", [])
            followup_questions = result.get("followup_questions", [])
            message_placeholder.markdown(response)
            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        content = source["content"] if isinstance(source, dict) and "content" in source else str(source)
                        st.markdown(content)
                        if isinstance(source, dict) and "metadata" in source and source["metadata"]:
                            try:
                                st.markdown(f"*Metadata: {json.dumps(source['metadata'], indent=2)}*")
                            except:
                                st.markdown(f"*Metadata: {str(source['metadata'])}*")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            st.session_state.followup_questions = followup_questions
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })


def render_upload():
    """Render the upload interface."""
    st.header("Upload Documents")

    upload_mode = st.radio("Select upload type:", ["static", "user"], format_func=lambda x: "General Insurance (Static)" if x=="static" else "User Policy (User-specific)")
    user_id = None
    if upload_mode == "user":
        user_id = st.text_input("Enter User ID for this policy upload:")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and (upload_mode == "static" or (upload_mode == "user" and user_id)):
        with st.spinner("Processing documents..."):
            index_documents(uploaded_files, upload_mode, user_id)
            st.success(f"Successfully processed {len(uploaded_files)} documents")
    elif uploaded_files and upload_mode == "user" and not user_id:
        st.warning("Please enter a User ID for user-specific upload.")


def index_documents(uploaded_files, mode, user_id=None):
    """Process and index uploaded PDF documents using agentic workflow."""
    try:
        st.info("Processing uploaded documents...")
        for file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(file.read())
            temp_file.close()
            try:
                chunk_count = ingest_document(mode, temp_file.name, user_id)
                st.success(f"Indexed {chunk_count} chunks from {file.name} in {mode} KB.")
            except Exception as e:
                st.error(f"Error ingesting {file.name}: {str(e)}")
            finally:
                os.unlink(temp_file.name)
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")


def render_scraping_sidebar():
    from src.rag.agentic_workflow import INSURER_URLS, scrape_and_ingest_pdfs_from_page
    st.sidebar.header("📄 Scrape Insurer PDFs")
    insurer = st.sidebar.selectbox("Select Insurance Company", list(INSURER_URLS.keys()))
    doc_types = st.sidebar.multiselect(
        "Select Document Types",
        ["Brochure", "Policy Wordings", "Benefits Guide"],
        default=["Brochure"]
    )
    if st.sidebar.button("Scrape & Ingest PDFs"):
        with st.sidebar:
            for doc_type in doc_types:
                url = INSURER_URLS[insurer].get(doc_type)
                if url:
                    st.info(f"Scraping {doc_type} from {insurer}...")
                    count = scrape_and_ingest_pdfs_from_page(url)
                    st.success(f"Ingested {count} PDFs from {doc_type} ({insurer})")
                else:
                    st.warning(f"No URL for {doc_type} in {insurer}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Agentic RAG",
        page_icon="📚",
        layout="wide"
    )
    
    render_scraping_sidebar()
    st.title("Agentic RAG")
    st.markdown("<p style='font-size:0.9em;color:gray;'>Created by Deep Podder</p>", unsafe_allow_html=True)
    
    # Initialize components if not already done
    if not st.session_state.initialized:
        if initialize_components():
            st.success("Successfully initialized Agentic RAG components!")
        else:
            st.error("Failed to initialize components. Please check the logs.")
            st.stop()
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_upload()
        
    with col2:
        render_chat()
    
    # Add footer with credits and links
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Center-aligned footer with GitHub and LinkedIn links
    footer_html = """
    <div style="text-align: center; padding: 20px 0px; margin-top: 30px;">
        <p style="font-size: 16px;"><b>Created and maintained by Deep Podder</b></p>
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="https://github.com/unknown-spec10" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30">
                GitHub
            </a>
            <a href="http://www.linkedin.com/in/deeppodder2005" target="_blank">
                <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="30" height="30">
                LinkedIn
            </a>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
