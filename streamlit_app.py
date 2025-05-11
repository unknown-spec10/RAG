"""Agentic RAG - Streamlit Frontend"""
import streamlit as st
import os
import sys
from typing import List, Dict, Any
import tempfile
import json
import time
import uuid

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

# Define configuration directly
APP_CONFIG = {
    "pdf_processor": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "embeddings": {
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
        "model_name": "mixtral-8x7b",
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

def initialize_components():
    """Initialize RAG components."""
    try:
        # Initialize file cache
        st.session_state.file_cache = cache_utils.FileCache("data/cache")
        
        # Initialize components in order
        st.session_state.pdf_parser = pdf_parser.PDFParser()
        st.session_state.text_chunker = text_chunker.TextChunker(
            chunk_size=st.session_state.config['pdf_processor']['chunk_size'],
            chunk_overlap=st.session_state.config['pdf_processor']['chunk_overlap']
        )
        st.session_state.embedding_model = embedding_model.EmbeddingModel(
            model_name=st.session_state.config['embeddings']['model_name']
        )

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
    """Process a user query."""
    if not st.session_state.initialized:
        st.error("Please upload and index some documents first.")
        return
        
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            with st.spinner("Searching knowledge base..."):
                result = st.session_state.agent.query(query)
            
            response = result["response"]
            sources = result.get("sources", [])
            followup_questions = result.get("followup_questions", [])
            
            message_placeholder.markdown(response)
            
            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        # Safely handle content which might be a string or dict
                        if isinstance(source, dict) and "content" in source:
                            content = source["content"]
                            if isinstance(content, str):
                                st.markdown(content)
                            else:
                                st.markdown(str(content))
                        else:
                            st.markdown(str(source))
                            
                        # Safely handle metadata
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
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            index_documents(uploaded_files)
            st.success(f"Successfully processed {len(uploaded_files)} documents")

def index_documents(uploaded_files):
    """Process and index uploaded PDF documents."""
    if not uploaded_files:
        return
    
    try:
        documents = []
        for file in uploaded_files:
            # Save file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(file.getvalue())
            temp_file.close()
            
            # Parse PDF
            try:
                text = st.session_state.pdf_parser.parse_file(temp_file.name)
                
                # Chunk text
                chunks_data = st.session_state.text_chunker.chunk_text(text)
                
                # Extract just the text strings from the chunk dictionaries
                chunks = [chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk) for chunk in chunks_data]
                
                # Create embeddings
                embeddings = st.session_state.embedding_model.embed_texts(chunks)
                
                # Create documents for vector DB
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    doc_id = f"{file.name}_{i}"
                    
                    # Ensure chunk is a string
                    if not isinstance(chunk, str):
                        chunk_text = str(chunk)
                    else:
                        chunk_text = chunk
                        
                    # Create document with proper metadata
                    documents.append({
                        "id": doc_id,
                        "text": chunk_text,
                        "embedding": embedding,
                        "metadata": {
                            "filename": file.name,
                            "page": i,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    })
                    st.session_state.indexed_files.add(doc_id)
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
        
        # Add documents to vector DB
        st.session_state.vector_db.add_documents(
            documents,
            embedding_key="embedding",
            text_key="text",
            metadata_key="metadata",
            ids=[doc["id"] for doc in documents]
        )
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Agentic RAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Agentic RAG")
    
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
