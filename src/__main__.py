"""Main entry point for the RAG system."""
import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import key components
from pdf_processor.pdf_parser import PDFParser
from pdf_processor.text_chunker import TextChunker
from agents.rag_agent import RAGAgent, MockLLM
from rag.chroma_retriever import ChromaRetriever

__all__ = ['PDFParser', 'TextChunker', 'RAGAgent', 'MockLLM', 'ChromaRetriever'] 