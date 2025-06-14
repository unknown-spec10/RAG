"""Streamlit app for RAG system."""
import streamlit as st
import os
from src.pdf_processor.pdf_parser import PDFParser
from src.pdf_processor.text_chunker import TextChunker
from src.agents.rag_agent import RAGAgent, MockLLM
from src.rag.retriever import RAGRetriever
from src.rag.embeddings import TFIDFEmbeddings
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
            
        # Initialize embeddings and retriever
        embeddings = TFIDFEmbeddings()
        retriever = RAGRetriever(embeddings=embeddings)
        
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
            text = pdf_parser.parse_file(tmp_path)
            
            # Create initial document
            document = {
                "text": text,
                "metadata": {"source": uploaded_file.name}
            }
            
            # Chunk the document
            chunker = TextChunker(
                chunk_size=1000,  # Smaller chunks to stay within token limits
                chunk_overlap=200,
                chunking_strategy="recursive"
            )
            
            chunked_docs = chunker.chunk_documents([document])
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
                    result = agent.query(query, chunked_docs)
                    
                    # Display response
                    st.write(result["response"])
                    
                    # Display follow-up questions
                    if result["followup_questions"]:
                        for i, question in enumerate(result["followup_questions"], 1):
                            st.write(f"{i}. {question}")
                            
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 