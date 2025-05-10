"""Streamlit frontend for the Agentic RAG application.
Created and maintained by Deep Podder
"""

# Fix for PyTorch module error in Streamlit
import sys
from types import ModuleType

try:
    # Create a custom module finder that blocks torch.classes
    class PyTorchClassBlocker:
        def __init__(self):
            self.original_import = __import__
        
        def custom_import(self, name, *args, **kwargs):
            # Block torch.classes access during streamlit's module scanning
            if name == 'torch.classes' or name.startswith('torch.classes.'):
                if 'streamlit.watcher' in sys.modules:
                    # Return a dummy module to prevent errors
                    return ModuleType(name)
            return self.original_import(name, *args, **kwargs)
        
        def install(self):
            __builtins__['__import__'] = self.custom_import
        
        def uninstall(self):
            __builtins__['__import__'] = self.original_import

    # Install the blocker
    blocker = PyTorchClassBlocker()
    blocker.install()
except Exception:
    # If the import hook can't be installed, continue anyway
    pass
import os
import sys
import streamlit as st
import tempfile
import json
import time
from typing import List, Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import project modules after setting up the path
from src.pdf_processor.text_chunker import TextChunker
from src.utils.cache import FileCache

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pdf_files = []
    st.session_state.indexed_files = set()
    st.session_state.messages = []
    st.session_state.followup_questions = []
    # Initialize file cache
    os.makedirs("data/cache", exist_ok=True)
    st.session_state.file_cache = FileCache("data/cache")

def initialize_components():
    """Initialize RAG components lazily."""
    if not st.session_state.initialized:
        from src.pdf_processor.pdf_parser import PDFParser
        from src.pdf_processor.text_chunker import TextChunker
        from src.embeddings.embedding_model import EmbeddingModel
        try:
            from src.vector_db.chroma_db import ChromaVectorDB
        except ImportError:
            raise ImportError("ChromaVectorDB is not defined or cannot be imported from 'src.vector_db.chroma_db'")
        # Verify that the ChromaVectorDB class exists in the specified module
        from src.rag.retriever import RAGRetriever
        from src.rag.context_protocol import ContextProtocolManager
        from src.agents.rag_agent import RAGAgent

        # Get the data directory - use .streamlit folder for cloud deployment
        data_dir = "data"
        # Create persistent directories
        for folder in ["pdfs", "chroma_db", "cache"]:
            os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

        # Initialize components
        st.session_state.pdf_parser = PDFParser(use_pdfplumber=False)
        st.session_state.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        st.session_state.embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
        
        try:
            # Use relative paths for better cloud compatibility
            chroma_persist_dir = os.path.join(data_dir, "chroma_db")
            st.session_state.vector_db = ChromaVectorDB(
                persist_directory=chroma_persist_dir,
                collection_name="documents"
            )
            st.session_state.context_protocol = ContextProtocolManager()
            st.session_state.retriever = RAGRetriever(
                vector_db=st.session_state.vector_db,
                embedding_model=st.session_state.embedding_model,
                top_k=5
            )
            # For Streamlit Cloud, we'll use an OpenAI model as default
            # since Ollama may not be available
            llm_name = "gpt-3.5-turbo"
            if os.environ.get("OPENAI_API_KEY") is None:
                llm_name = "llama3"  # Default model option
                
            st.session_state.agent = RAGAgent(
                retriever=st.session_state.retriever,
                model_name=llm_name,
                context_protocol=st.session_state.context_protocol
            )
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return False
    return True

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
                        st.markdown(source["content"])
                        if "metadata" in source and source["metadata"]:
                            st.markdown(f"*Metadata: {json.dumps(source['metadata'], indent=2)}*")
            
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
    """Render the upload and management interface."""
    st.header("Upload and Manage Documents")
    
    # Upload PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.form(key="index_form"):
            st.write("Configure Indexing")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks in characters"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between consecutive chunks in characters"
            )
            
            index_button = st.form_submit_button("Index Documents")
            if index_button:
                index_documents(uploaded_files, chunk_size, chunk_overlap)
    
    # Display indexed files
    if st.session_state.indexed_files:
        st.subheader("Indexed Documents")
        for file_path in st.session_state.indexed_files:
            st.text(os.path.basename(file_path))
    
    # Database information
    if st.session_state.initialized:
        st.subheader("Database Information")
        try:
            doc_count = st.session_state.vector_db.count()
            st.metric("Indexed Documents", doc_count)
            
            if st.button("Clear Database"):
                st.session_state.vector_db.client.reset()
                st.session_state.vector_db = ChromaVectorDB(
                    persist_directory=os.path.abspath("data/chroma_db"),
                    collection_name="documents"
                )
                st.session_state.indexed_files = set()
                st.success("Database cleared successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")

def index_documents(uploaded_files, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process and index uploaded PDF documents."""
    if not initialize_components():
        return
        
    st.session_state.text_chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    cached_count = 0
    indexed_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i / total_files)
        progress_bar.progress(progress)
        
        # Save the uploaded file to the pdfs directory
        save_path = os.path.join("data/pdfs", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Check if the file is already in cache with the same chunking parameters
        if st.session_state.file_cache.is_file_cached(save_path, chunk_size, chunk_overlap):
            status_text.text(f"Using cached version of {uploaded_file.name}... ({i+1}/{total_files})")
            # Add to indexed files set
            st.session_state.indexed_files.add(save_path)
            cached_count += 1
            continue
            
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
        
        # Process and index the file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            documents = st.session_state.pdf_parser.parse_pdf(pdf_path)
            metadata = st.session_state.pdf_parser.extract_metadata(pdf_path)
            
            for doc in documents:
                doc["metadata"].update({
                    "file_name": uploaded_file.name,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                })
            
            chunked_documents = st.session_state.text_chunker.chunk_documents(documents)
            embedded_documents = st.session_state.embedding_model.embed_documents(chunked_documents)
            
            # Add documents to vector DB and get their IDs
            doc_ids = st.session_state.vector_db.add_documents(embedded_documents)
            
            # Add to cache to avoid reprocessing in the future
            st.session_state.file_cache.add_file_to_cache(
                save_path, chunk_size, chunk_overlap, doc_ids
            )
            
            st.session_state.indexed_files.add(save_path)
            indexed_count += 1
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            os.unlink(pdf_path)
    
    progress_bar.progress(1.0)
    status_text.text("Indexing complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    if cached_count > 0 and indexed_count > 0:
        st.success(f"Successfully processed {total_files} documents! ({indexed_count} newly indexed, {cached_count} from cache)")
    elif cached_count > 0:
        st.success(f"All {cached_count} documents were already indexed previously and loaded from cache!")
    else:
        st.success(f"Successfully indexed {indexed_count} documents!")

def main():
    """Main application entry point."""
    st.title("📚 Agentic RAG Knowledge Base")
    st.caption("Created and maintained by Deep Podder")
    
    tab1, tab2, tab3 = st.tabs(["Chat", "Upload & Manage", "Cache Info"])
    
    with tab1:
        render_chat()
    with tab2:
        render_upload()
    with tab3:
        render_cache_info()
        
def render_cache_info():
    """Render information about cached files."""
    st.header("Cache Information")
    
    if not st.session_state.initialized:
        initialize_components()
    
    cached_files = st.session_state.file_cache.get_cached_files()
    
    if not cached_files:
        st.info("No files are currently cached.")
        return
    
    st.write(f"Total cached files: {len(cached_files)}")
    
    for file_info in cached_files:
        with st.expander(file_info['filename']):
            st.write(f"**Path:** {file_info['path']}")
            st.write(f"**Chunks:** {file_info['document_count']}")
            st.write(f"**Chunk Size:** {file_info['chunk_size']}")
            st.write(f"**Chunk Overlap:** {file_info['chunk_overlap']}")
            st.write(f"**Indexed On:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_info['timestamp']))}")
            
            if st.button(f"Remove from cache", key=f"remove_{file_info['path']}"):
                st.session_state.file_cache.remove_file_from_cache(file_info['path'])
                st.success(f"Removed {file_info['filename']} from cache!")
                st.rerun()
    
    if st.button("Clear All Cache"):
        st.session_state.file_cache.clear_cache()
        st.success("Cache cleared successfully!")
        st.rerun()

if __name__ == "__main__":
    main()
