"""
Agentic RAG workflow for PDF processing and question answering.
"""
import os
from typing import Optional, List, Dict, Any
from src.pdf_processor.pdf_parser import PDFParser
from src.pdf_processor.text_chunker import TextChunker
from src.rag.retriever import RAGRetriever

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a PDF file and return its chunks.
    Args:
        file_path: path to PDF file
    Returns:
        List of document chunks with text and metadata
    """
    parser = PDFParser()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)

    text = parser.parse_file(file_path)
    chunks = chunker.chunk_text(text)
    
    # Add metadata to chunks
    documents = []
    for chunk in chunks:
        documents.append({
            "text": chunk["text"],
            "metadata": {"source": os.path.basename(file_path)}
        })
    
    return documents

def retrieve_context(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve context from documents for a query.
    Args:
        query: user query
        documents: list of document chunks
        top_k: number of documents to retrieve
    Returns:
        List of relevant document chunks
    """
    retriever = RAGRetriever(top_k=top_k)
    return retriever.retrieve(query, documents)

