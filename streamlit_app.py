"""Streamlit app for RAG system."""
import streamlit as st
import os
import sys
import logging
import tempfile
from typing import Optional, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some verbose loggers
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Ensure src/ is on sys.path for import resolution
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.rag.pdf_processor.pdf_parser import PDFParser
    from src.pdf_processor.text_chunker import TextChunker
    from src.agents.rag_agent import RAGAgent, MockLLM
    from src.rag.chroma_retriever import ChromaRetriever
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    st.error("Error loading required modules. Please check the application logs.")
    st.stop()

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

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

def get_serp_api_key() -> Optional[str]:
    """Get SERP API key from secrets with proper error handling."""
    try:
        serp_key = st.secrets.get("serp_api_key")
        if not serp_key:
            st.warning("SERP API key not found. Web search functionality will be disabled.")
            return None
        return serp_key
    except Exception as e:
        logger.warning(f"Error accessing SERP API key: {str(e)}")
        return None

def initialize_agent(use_reranker=True):
    """Initialize the RAG agent with proper error handling."""
    try:
        # Get API keys
        api_key = get_api_key()
        if not api_key:
            st.error("Failed to get API key. Please check your Streamlit Cloud secrets.")
            return None
            
        serp_api_key = get_serp_api_key()
        
        # Initialize retriever with ChromaDB
        try:
            retriever = ChromaRetriever()
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            st.error("Failed to initialize document database. Please try again.")
            return None
        
        # Initialize agent with SERP API key
        try:
            agent = RAGAgent(
                retriever=retriever,
                api_key=api_key,
                use_reranker=use_reranker,
                serp_api_key=serp_api_key
            )
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {str(e)}")
            st.error("Failed to initialize AI agent. Please try again.")
            return None
        
        # Store the LLM type in session state
        st.session_state.using_mock_llm = isinstance(agent.llm, MockLLM)
        
        return agent
    except Exception as e:
        logger.error(f"Error in agent initialization: {str(e)}")
        st.error("An unexpected error occurred during initialization. Please try again.")
        return None

def process_pdf_file(uploaded_file, chunking_strategy, chunk_size, chunk_overlap):
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
            
            # Add source information to each document and clean metadata
            for doc in documents:
                # Ensure metadata exists
                if 'metadata' not in doc:
                    doc['metadata'] = {}
                
                # Add source information
                doc['metadata']['source'] = uploaded_file.name
                
                # Clean metadata to ensure no None values
                cleaned_metadata = {}
                for key, value in doc['metadata'].items():
                    if value is None:
                        cleaned_metadata[key] = ""
                    elif isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                doc['metadata'] = cleaned_metadata
            
            # Chunk the documents with error handling
            try:
                chunker = TextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunking_strategy=chunking_strategy
                )
                chunked_docs = chunker.chunk_documents(documents)
                return chunked_docs
            except Exception as chunking_error:
                logger.error(f"Error during chunking with strategy '{chunking_strategy}': {str(chunking_error)}")
                # Fallback to recursive chunking
                st.warning(f"Chunking strategy '{chunking_strategy}' failed, falling back to recursive chunking.")
                chunker = TextChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
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

def get_user_consent_for_web_search(analysis: Dict[str, Any]) -> bool:
    """
    Check if user has consented to web search via sidebar checkbox.
    
    Args:
        analysis: Query analysis from orchestrator
        
    Returns:
        True if user consents to web search, False otherwise
    """
    # Check if web search is allowed via sidebar checkbox
    return st.session_state.get('allow_web_search', False)

def process_query_with_orchestration(agent, query: str, allow_web_search: bool):
    """Process query using orchestration with user consent from sidebar."""
    
    def consent_callback(analysis):
        if allow_web_search:
            # Show brief info about the web search decision
            if analysis['strategy']['use_web_search']:
                with st.expander("🌐 Web Search Info", expanded=False):
                    st.info(f"**Decision:** {analysis['reasoning']}")
                    if analysis.get('filtered_query'):
                        st.write(f"**Filtered Query:** `{analysis['filtered_query']}`")
                        st.caption("*(Sensitive information removed for privacy)*")
            return True
        else:
            return False
    
    try:
        # Use orchestrated query processing
        result = agent.query_with_orchestration(query, user_consent_callback=consent_callback)
        
        # Display the enhanced response
        display_orchestrated_response(result)
        
    except Exception as e:
        logger.error(f"Error in orchestrated query processing: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
        
        # Fallback to regular query processing
        st.info("Falling back to local-only search...")
        try:
            fallback_result = agent.query(query)
            display_response(fallback_result)
        except Exception as fallback_error:
            logger.error(f"Fallback query also failed: {str(fallback_error)}")
            st.error(f"Query processing failed: {str(fallback_error)}")

def display_orchestrated_response(result: Dict[str, Any]):
    """Display the orchestrated response with clean, simple interface."""
    
    # Check if web search was used
    query_analysis = result.get("query_analysis", {})
    web_results_count = query_analysis.get("web_results_count", 0)
    
    # Show web search warning if applicable
    if web_results_count > 0:
        st.warning("⚠️ **Web Search Results Included** - This response contains information from web search results. Please verify important information from reliable sources.")
    
    # Display the main response
    st.write("### 🤖 Response")
    st.write(result.get("response", "No response available"))
    
    # Show confidence score if present
    if "confidence" in result:
        st.info(f"**LLM Confidence Score:** {result['confidence']:.2f}")
    
    # Display sources with type indicators (simplified)
    st.write("### 📚 Sources")
    sources = result.get("sources", [])
    
    if sources:
        for i, source in enumerate(sources, 1):
            source_type = source.get("type", "unknown")
            icon = "📄" if source_type == "local" else "🌐" if source_type == "web" else "📋"
            
            with st.expander(f"{icon} Source {i}: {source.get('source', 'Unknown')}"):
                st.write(source.get("text", "No text available"))
    else:
        st.info("No sources found.")
    
    # Display follow-up questions
    if result.get("followup_questions"):
        st.write("### ❓ Suggested Follow-up Questions")
        for i, question in enumerate(result["followup_questions"], 1):
            st.write(f"{i}. {question}")

def display_response(result):
    """Display the response with sources and follow-up questions."""
    # Display the response
    st.write("### Response")
    st.write(result.get("response", "No response available"))
    # Show confidence score if present
    if "confidence" in result:
        st.info(f"**LLM Confidence Score:** {result['confidence']:.2f}")
    
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
    st.title("Sasthya Samadhan :)")
    st.markdown("""
    <style>
    .big-font {font-size:20px !important;}
    </style>
    """, unsafe_allow_html=True)

    # Session reset
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    # Chunking controls
    st.sidebar.header("Chunking Options")
    chunking_strategy = st.sidebar.selectbox("Chunking Strategy", ["recursive", "semantic"], index=0)
    chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    # Web search options
    st.sidebar.markdown("---")
    st.sidebar.header("Search Options")
    use_orchestration = st.sidebar.checkbox("Enable Intelligent Search Strategy", value=True, 
                                           help="Uses AI to decide when to search local documents vs web")
    
    # Web search consent checkbox
    allow_web_search = st.sidebar.checkbox("Allow Web Search", value=False,
                                         help="Enable web search for current events and public information. Your query will be filtered to remove sensitive data.")
    
    if use_orchestration and allow_web_search:
        st.sidebar.success("🌐 Web search enabled")
        st.sidebar.caption("Queries will be filtered for privacy before web search")
    elif use_orchestration:
        st.sidebar.info("📄 Local search only")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Created and maintained by Deep Podder")

    # Initialize the agent
    agent = initialize_agent()
    if agent is None:
        st.stop()

    # File upload with size limit
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=['pdf'],
        help=f"Maximum file size: {MAX_FILE_SIZE/1024/1024}MB",
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        uploaded_file.seek(0, 2)
        file_size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)
        if file_size_mb > 50:
            st.error(f"File is too large. Maximum allowed size is 50MB. Your file is {file_size_mb:.2f}MB.")
        else:
            try:
                # Process the PDF and get chunked documents
                chunked_docs = process_pdf_file(uploaded_file, chunking_strategy, chunk_size, chunk_overlap)
                st.success(f"Document processed successfully! Created {len(chunked_docs)} chunks.")
                agent.retriever.add_documents(chunked_docs)
                # Query input
                query = st.text_input("Ask a question about the document:")
                if query:
                    # Store web search preference
                    st.session_state['allow_web_search'] = allow_web_search
                    
                    if use_orchestration:
                        # Use orchestrated query processing with sidebar setting
                        process_query_with_orchestration(agent, query, allow_web_search)
                    else:
                        # Use traditional local-only processing
                        try:
                            result = agent.query(query)
                            display_response(result)
                        except Exception as e:
                            logger.error(f"Error processing query: {str(e)}")
                            st.error(f"Error processing query: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                st.error(f"Error processing file: {str(e)}")
    
    # credits
    st.markdown("---")
    st.markdown("----Created and maintained by Deep Podder----")

if __name__ == "__main__":
    main() 