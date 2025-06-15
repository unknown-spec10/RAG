"""Text chunking utilities for RAG system."""
from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ParagraphChunker:
    """Chunks text into paragraphs while preserving semantic meaning."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the paragraph chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for paragraph-based splitting
        self.separators = separators or [
            "\n\n",  # Double newline
            "\n",    # Single newline
            ". ",    # Period followed by space
            "! ",    # Exclamation mark followed by space
            "? ",    # Question mark followed by space
            "; ",    # Semicolon followed by space
            ": ",    # Colon followed by space
            ", ",    # Comma followed by space
            " ",     # Space
            ""       # No separator (fallback)
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs and other natural boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Post-process chunks to ensure they end at sentence boundaries
        processed_chunks = []
        for chunk in chunks:
            # Find the last sentence boundary
            last_period = chunk.rfind(". ")
            last_exclamation = chunk.rfind("! ")
            last_question = chunk.rfind("? ")
            
            # Get the last boundary position
            last_boundary = max(last_period, last_exclamation, last_question)
            
            if last_boundary > 0:
                # Split at the last sentence boundary
                processed_chunks.append(chunk[:last_boundary + 1])
                # Add the remaining text to the next chunk
                if last_boundary + 1 < len(chunk):
                    processed_chunks.append(chunk[last_boundary + 1:])
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the input text by removing extra whitespace and normalizing line endings.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        
        return text.strip()
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            # Get the text content
            text = doc.get('text', '')
            if not text:
                continue
                
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = doc.copy()
                chunked_doc['text'] = chunk
                chunked_doc['chunk_id'] = f"{doc.get('id', '')}_chunk_{i}"
                chunked_docs.append(chunked_doc)
        
        return chunked_docs 