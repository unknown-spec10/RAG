"""Text chunking module for splitting text into chunks for embedding."""
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


class TextChunker:
    """Class for chunking text into manageable pieces for embedding."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive"
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
            chunking_strategy: Strategy for chunking text. Options: "recursive", "markdown".
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        if chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif chunking_strategy == "markdown":
            # For markdown documents with headers
            headers_to_split_on = [
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
                ("####", "header4"),
            ]
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of dictionaries, each containing a chunk of text and its metadata.
        """
        if self.chunking_strategy == "markdown":
            # First split by headers
            md_header_splits = self.markdown_splitter.split_text(text)
            # Then split further by size
            chunks = []
            for md_doc in md_header_splits:
                sub_chunks = self.text_splitter.create_documents([md_doc.page_content])
                for i, chunk in enumerate(sub_chunks):
                    metadata = md_doc.metadata.copy()
                    metadata["chunk_id"] = i
                    chunks.append({
                        "text": chunk.page_content,
                        "metadata": metadata
                    })
            return chunks
        else:
            # Standard recursive splitting
            sub_texts = self.text_splitter.split_text(text)
            chunks = [
                {
                    "text": sub_text,
                    "metadata": {
                        "chunk_id": i,
                        "chunk_size": len(sub_text),
                    }
                }
                for i, sub_text in enumerate(sub_texts)
            ]
            return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of document texts into chunks.
        
        Args:
            documents: List of documents to chunk. Each should have a 'text' field.
            
        Returns:
            A list of dictionaries, each containing a chunk of text and its metadata.
        """
        results = []
        
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            metadata = doc.get("metadata", {})
            page_num = doc.get("page_num", None)
            
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    **chunk.get("metadata", {}),
                    "page_num": page_num,
                    "chunk_index": i,
                }
                
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk_metadata
                })
        
        return results
