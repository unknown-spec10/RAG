"""Agentic RAG - Streamlit Frontend

A Retrieval-Augmented Generation (RAG) application that allows you to chat with your documents.
Created by Deep Podder.
"""
import streamlit as st
import os
import sys
from typing import List, Dict, Any
import tempfile
import json
import time
import uuid
import numpy as np  # For array operations with embeddings

# Set page title and favicon
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize the app state if it doesn't exist
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents = []
    st.session_state.question_history = []
    st.session_state.answer_history = []
    st.session_state.sources_history = []
    st.session_state.followup_questions = []
    st.session_state.document_count = 0
    st.session_state.error = None
    # Create data directories
    os.makedirs("data/cache", exist_ok=True)
    # Initialize configuration
    st.session_state.config = APP_CONFIG

def init_api_keys():
    """Initialize API keys from environment variables or Streamlit secrets."""
    # Groq API Key (required)
    groq_api_key = None
    
    # Try to get from Streamlit secrets first
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    # Then try environment variables
    elif os.environ.get("GROQ_API_KEY"):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        
    # Check if we have a valid Groq API key
    if not groq_api_key:
        st.error("❌ Groq API key not found! Please add it to your secrets or environment variables.")
        st.info("To add your Groq API key, create a .streamlit/secrets.toml file with:\n\nGROQ_API_KEY = \"your-groq-api-key-here\"")
        return False
        
    # Google API Key (optional)
    google_api_key = None
    
    # Try to get from Streamlit secrets first
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    # Then try environment variables
    elif os.environ.get("GOOGLE_API_KEY"):
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        
    # Store the API keys in environment variables for modules to access
    os.environ["GROQ_API_KEY"] = groq_api_key
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.info("ℹ️ Google API key not found. Using fallback hash-based embeddings.")
    
    return True

def initialize_components():
    """Initialize RAG components."""
    try:
        # Get API keys first
        if not init_api_keys():
            return False
        
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
    try:
        documents = []
        st.info("Processing uploaded documents...")
        
        for file in uploaded_files:
            # Create a unique ID based on filename and timestamp
            file_id = uuid.uuid4().hex
            st.info(f"Processing file: {file.name}")
            
            # Create a temporary file to save the uploaded file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(file.read())
            temp_file.close()
            
            try:
                # Parse the PDF file
                text = st.session_state.pdf_parser.parse_file(temp_file.name)
                st.info(f"Extracted text from {file.name}")
                # Since parse_file returns a single string, we'll treat it as one page
                text_by_page = [text]
                
                # Process each page
                for i, text in enumerate(text_by_page):
                    if not text.strip():  # Skip empty pages
                        continue
                        
                    # Create document ID using filename, page number and UUID
                    doc_id = f"{file.name.split('.')[0]}_page_{i}_{file_id}"
                    
                    # Skip if we've already indexed this document
                    if doc_id in st.session_state.indexed_files:
                        continue
                    
                    # Split text into chunks
                    chunks_data = st.session_state.text_chunker.chunk_text(text)
                    st.info(f"Created {len(chunks_data)} chunks from page {i}")
                    
                    # Extract just the text strings from the chunk dictionaries
                    chunks = [chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk) for chunk in chunks_data]
                    
                    # Create embeddings using hash-based method (simplest and most reliable)
                    embeddings = []
                    # Determine the required embedding dimension
                    embedding_dim = 384  # Hardcoded to match vector DB expectation
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            # Generate embedding directly from text - always 384 dimensions
                            embedding = st.session_state.embedding_model._generate_hash_embedding(chunk)
                            
                            # Double-check dimension
                            if embedding.shape != (embedding_dim,):
                                st.warning(f"Fixing embedding dimension for chunk {chunk_idx}: got {embedding.shape}, expected ({embedding_dim},)")
                                # Resize or create new embedding with correct dimension
                                fixed_embedding = np.zeros(embedding_dim, dtype=np.float32)
                                # Copy values from original embedding up to min length
                                min_dim = min(len(embedding), embedding_dim)
                                fixed_embedding[:min_dim] = embedding[:min_dim]
                                # Normalize
                                norm = np.linalg.norm(fixed_embedding)
                                if norm > 0:
                                    fixed_embedding /= norm
                                embedding = fixed_embedding
                                
                            embeddings.append(embedding)
                        except Exception as emb_err:
                            st.error(f"Error generating embedding for chunk {chunk_idx}: {str(emb_err)}")
                            # Create a simple zero embedding as fallback with correct dimension
                            embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
                    
                    # Convert to numpy array for vector DB
                    embeddings = np.array(embeddings)
                    st.info(f"Created {len(embeddings)} embeddings for page {i}")
                    
                    # Create documents for vector DB
                    for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_id = f"{doc_id}_chunk_{j}"
                        documents.append({
                            "id": chunk_id,
                            "text": chunk,
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
        
        # Add documents to vector DB with additional error handling
        try:
            if not documents:
                st.warning("No documents were processed. Please try uploading a different PDF.")
                return
                
            st.info(f"Adding {len(documents)} document chunks to vector database")
            
            # Check that embeddings have the right shape and fix if needed
            embedding_dim = 384  # Hardcoded to ensure consistency
            dimension_fixes = 0
            
            for doc in documents:
                if doc["embedding"].shape != (embedding_dim,):
                    # Reshape or fix the embedding
                    dimension_fixes += 1
                    fixed_embedding = np.zeros(embedding_dim, dtype=np.float32)
                    # Copy values from original embedding up to min length
                    min_dim = min(len(doc["embedding"]), embedding_dim)
                    fixed_embedding[:min_dim] = doc["embedding"][:min_dim]
                    # Normalize
                    norm = np.linalg.norm(fixed_embedding)
                    if norm > 0:
                        fixed_embedding /= norm
                    doc["embedding"] = fixed_embedding
            
            if dimension_fixes > 0:
                st.warning(f"Fixed dimensions for {dimension_fixes} embeddings to match vector DB requirements ({embedding_dim} dimensions)")
            
            # Add to vector DB
            st.session_state.vector_db.add_documents(
                documents,
                embedding_key="embedding",
                text_key="text",
                metadata_key="metadata",
                ids=[doc["id"] for doc in documents]
            )
            
            st.success(f"Successfully indexed {len(documents)} chunks from {len(uploaded_files)} documents!")
            
        except Exception as db_err:
            st.error(f"Error adding documents to vector database: {str(db_err)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

def process_documents(uploaded_files):
    """Process uploaded PDF documents."""
    if not uploaded_files:
        return False
        
    processed_count = 0
    
    with st.spinner("Processing documents..."):
        try:
            # Initialize the PDF parser
            pdf_parser_instance = pdf_parser.PDFParser()
            
            # Initialize the text chunker
            text_chunker_instance = text_chunker.TextChunker(
                chunk_size=APP_CONFIG["pdf_processor"]["chunk_size"],
                chunk_overlap=APP_CONFIG["pdf_processor"]["chunk_overlap"]
            )
            
            # Initialize embedding model
            embedding_model_instance = embedding_model.EmbeddingModel()
            
            # Initialize vector database
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            vector_db_instance = faiss_db.FAISSVectorDB(
                persist_directory="data/cache",
                collection_name="documents",
                dimension=384  # Using 384 dimensions for consistency
            )
            
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                try:
                    # Show progress message
                    st.text(f"Processing: {uploaded_file.name}")
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extract text from PDF
                    try:
                        extracted_text = pdf_parser_instance.parse_file(tmp_path)
                    except Exception as e:
                        st.error(f"Error parsing PDF {uploaded_file.name}: {str(e)}")
                        continue
                        
                    # Remove the temporary file
                    os.unlink(tmp_path)
                    
                    # Split text into chunks
                    chunks = text_chunker_instance.split_text(extracted_text)
                    
                    if not chunks:
                        st.warning(f"No content extracted from {uploaded_file.name}")
                        continue
                    
                    st.text(f"Created {len(chunks)} chunks from {uploaded_file.name}")
                    
                    # Create documents with metadata
                    documents = []
                    for i, chunk in enumerate(chunks):
                        # Generate embedding for the chunk
                        try:
                            # Generate embedding
                            embedding = embedding_model_instance.get_embedding(chunk)
                        except Exception as e:
                            st.warning(f"Using fallback embedding method")
                            # Create a simple fallback embedding
                            embedding = embedding_model_instance._generate_hash_embedding(chunk)
                        
                        # Create document
                        doc = {
                            "text": chunk,
                            "embedding": embedding,
                            "metadata": {
                                "source": uploaded_file.name,
                                "chunk": i,
                                "total_chunks": len(chunks)
                            }
                        }
                        documents.append(doc)
                    
                    # Add to vector database
                    vector_db_instance.add_documents(documents)
                    
                    # Add to session state
                    file_info = {
                        "name": uploaded_file.name,
                        "chunks": len(chunks),
                        "embedding_dim": len(embedding) if len(documents) > 0 else 0
                    }
                    st.session_state.documents.append(file_info)
                    
                    processed_count += 1
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
                    st.session_state.error = error_msg
                    print(error_msg)
                    print(traceback.format_exc())
                    st.error(error_msg)
            
            if processed_count > 0:
                st.session_state.document_count += processed_count
                st.success(f"✅ Successfully processed {processed_count} documents.")
                return True
            else:
                st.warning("⚠️ No documents were processed successfully.")
                return False
                
        except Exception as e:
            import traceback
            error_msg = f"Error in document processing: {str(e)}"
            st.session_state.error = error_msg
            print(error_msg)
            print(traceback.format_exc())
            st.error(error_msg)
            return False

def main():
    """Main application function."""
    # Set title and banner
    st.title("📚 Agentic RAG")
    st.subheader("Chat with your documents using AI")
    
    # Add creator credit
    st.markdown("<div style='text-align: right; color: gray; padding-bottom: 10px;'>Created by Deep Podder</div>", unsafe_allow_html=True)
    
    # Initialize API keys
    if not init_api_keys():
        st.stop()
    
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
