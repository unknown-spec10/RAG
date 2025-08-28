"""
Document ingestion utility for adding documents to both ChromaRetriever and RAGRetriever (TF-IDF).
Usage:
    from src.rag.chroma_retriever import ChromaRetriever
    from src.rag.rag.retriever import RAGRetriever
    from src.rag.pdf_processor.text_chunker import TextChunker
    from src.rag.document_ingest import ingest_documents
    
    # Assume you have a list of raw docs (with 'text' and 'metadata')
    chunker = TextChunker()
    chunks = chunker.chunk_documents(raw_docs)
    ingest_documents(chunks, chroma_retriever, rag_retriever)
"""
from typing import List, Dict, Any

def ingest_documents(
    documents: List[Dict[str, Any]],
    chroma_retriever,
    rag_retriever
):
    """
    Add documents to both ChromaRetriever and RAGRetriever.
    Args:
        documents: List of chunked documents (with 'text' and 'metadata')
        chroma_retriever: Instance of ChromaRetriever
        rag_retriever: Instance of RAGRetriever
    """
    chroma_retriever.add_documents(documents)
    rag_retriever.add_documents(documents)
