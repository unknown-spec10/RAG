"""Streamlit app for RAG system."""
import streamlit as st
import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import from src directory
from rag.pdf_processor.pdf_parser import PDFParser
from rag.pdf_processor.text_chunker import TextChunker
from rag.agents.rag_agent import RAGAgent, MockLLM
from rag.rag.chroma_retriever import ChromaRetriever
import logging
import tempfile
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB (matching config.toml)

# Initialize session state
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None
if 'chunked_docs' not in st.session_state:
    st.session_state.chunked_docs = []
if 'agent' not in st.session_state:
    st.session_state.agent = None

def get_api_key() -> Optional[str]:
    """Get API key from secrets with proper error handling."""
    try:
        api_key = st.secrets.get("groq_api_key")
        if not api_key:
            st.error("Groq API key not found in secrets. Please configure it in Streamlit Cloud secrets.")
            return None
        if not api_key.startswith("gsk_"):
            st.error("Invalid Groq API key format. It should start with 'gsk_'")
            return None
        return api_key
    except Exception as e:
        logger.error(f"Error accessing secrets: {str(e)}")
        st.error("Error accessing API key. Please check your Streamlit Cloud configuration.")
        return None

def initialize_agent():
    """Initialize the RAG agent with proper error handling."""
    try:
        # Get API key
        api_key = get_api_key()
        if not api_key:
            return None
            
        # Initialize retriever with ChromaDB
        retriever = ChromaRetriever()
        
        # Initialize agent
        agent = RAGAgent(
            retriever=retriever,
            api_key=api_key
        )
        
        # Store the LLM type in session state
        st.session_state.using_mock_llm = isinstance(agent.llm, MockLLM)
        
        return agent
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error(f"Error initializing agent: {str(e)}")
        return None

def process_pdf_file(uploaded_file):
    """Process uploaded PDF file and return chunked documents."""
    try:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB")
            
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Parse PDF
            pdf_parser = PDFParser()
            documents = pdf_parser.parse_file(tmp_path)
            
            # Add source information to each document
            for doc in documents:
                doc['metadata']['source'] = uploaded_file.name
            
            # Chunk the documents
            chunker = TextChunker(
                chunk_size=2000,  # Increased chunk size
                chunk_overlap=400,  # Increased overlap
                chunking_strategy="recursive"
            )
            
            chunked_docs = chunker.chunk_documents(documents)
            return chunked_docs
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def display_response(result):
    """Display the response with sources and follow-up questions."""
    # Display the response
    st.write("### Response")
    st.write(result.get("response", "No response available"))
    
    # Display sources with specific chunks
    st.write("### Supporting Information")
    sources = result.get("sources", [])
    for source in sources:
        with st.expander(f"Source: '{source.get('source', 'Unknown')}' (Page {source.get('page', 'Unknown')})"):
            st.write(source.get("text", "No text available"))
    
    # Display follow-up questions
    st.write("### Suggested Follow-up Questions")
    followup_questions = result.get("followup_questions", [])
    for i, question in enumerate(followup_questions, 1):
        st.write(f"{i}. {question}")

def main():
    st.title("Document Q&A System")
    
    # Initialize the agent
    agent = initialize_agent()
    if agent is None:
        st.stop()
    
    # File upload with size limit
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=['pdf'],
        help=f"Maximum file size: {MAX_FILE_SIZE/1024/1024}MB"
    )
    
    if uploaded_file is not None:
        try:
            # Process the PDF and get chunked documents
            chunked_docs = process_pdf_file(uploaded_file)
            
            # Add documents to retriever
            agent.retriever.add_documents(chunked_docs)
            
            st.success(f"Document processed successfully! Created {len(chunked_docs)} chunks.")
            
            # Query input
            query = st.text_input("Ask a question about the document:")
            
            if query:
                try:
                    # Process the query
                    result = agent.query(query)
                    
                    # Display response
                    display_response(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 