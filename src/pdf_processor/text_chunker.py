"""Text chunking module for splitting text into chunks for embedding."""
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


import re
from sentence_transformers import SentenceTransformer, util

class TextChunker:
    """Class for chunking text into manageable pieces for embedding."""
    
    def __init__(
        self, 
        chunk_size: int = 2000,  # Increased chunk size
        chunk_overlap: int = 400,  # Increased overlap
        chunking_strategy: str = "recursive"
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
            chunking_strategy: Strategy for chunking text. Options: "recursive", "markdown", "semantic".
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        # Initialize semantic model for semantic chunking
        if chunking_strategy == "semantic":
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load semantic model, falling back to recursive chunking: {e}")
                self.chunking_strategy = "recursive"
        
        if chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # More granular separators
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
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")
    
    def semantic_chunk_text(self, text: str, metadata: Dict[str, Any] = None, sim_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Split text into semantically coherent chunks using embedding similarity and paragraph/topic boundaries.
        Args:
            text: The text to split.
            metadata: Metadata to attach to each chunk.
            sim_threshold: Cosine similarity threshold for topic change.
        Returns:
            List of chunk dicts with enriched metadata.
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if len(paragraphs) <= 1:
            return [{"text": text, "metadata": metadata or {}}]
        # Compute embeddings
        embeddings = self.semantic_model.encode(paragraphs)
        chunks = []
        current_chunk = paragraphs[0]
        current_meta = metadata.copy() if metadata else {}
        for i in range(1, len(paragraphs)):
            sim = float(util.cos_sim(embeddings[i-1], embeddings[i]))
            if sim < sim_threshold or len(current_chunk) > self.chunk_size:
                # New chunk
                chunks.append({
                    "text": current_chunk,
                    "metadata": {**current_meta, "chunk_id": len(chunks), "semantic": True}
                })
                current_chunk = paragraphs[i]
            else:
                current_chunk += "\n\n" + paragraphs[i]
        # Add last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": {**current_meta, "chunk_id": len(chunks), "semantic": True}
            })
        return chunks

    def recursive_chunk_text(self, text: str, metadata: Dict[str, Any] = None, max_depth: int = 2, depth: int = 0, parent_id: str = None) -> List[Dict[str, Any]]:
        """
        Recursively split text into hierarchical chunks, adding hierarchy metadata.
        Args:
            text: The text to split.
            metadata: Metadata to attach.
            max_depth: Maximum recursion depth.
            depth: Current recursion level.
            parent_id: Parent chunk id.
        Returns:
            List of chunk dicts with hierarchy metadata.
        """
        if depth >= max_depth or len(text) <= self.chunk_size:
            return [{"text": text, "metadata": {**(metadata or {}), "chunk_id": 0, "depth": depth, "parent_id": parent_id}}]
        sub_texts = self.text_splitter.split_text(text)
        results = []
        for i, sub_text in enumerate(sub_texts):
            chunk_id = f"{parent_id or 'root'}_{i}"
            child_chunks = self.recursive_chunk_text(sub_text, metadata, max_depth, depth+1, chunk_id)
            for chunk in child_chunks:
                chunk["metadata"].update({"chunk_id": chunk_id, "depth": depth, "parent_id": parent_id})
                results.append(chunk)
        return results

    def enrich_metadata(self, chunk: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach richer metadata to a chunk.
        Args:
            chunk: The chunk dict.
            extra: Additional metadata fields.
        Returns:
            Updated chunk dict.
        """
        chunk["metadata"].update(extra)
        return chunk

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Flexible chunking with support for semantic, recursive, and legacy strategies.
        Args:
            text: The text to split.
            metadata: Metadata to attach.
            strategy: Override chunking strategy.
        Returns:
            List of chunk dicts.
        """
        strategy = strategy or self.chunking_strategy
        if strategy == "semantic":
            return self.semantic_chunk_text(text, metadata)
        elif strategy == "recursive":
            return self.recursive_chunk_text(text, metadata)
        elif strategy == "markdown":
            # First split by headers
            md_header_splits = self.markdown_splitter.split_text(text)
            # Then split further by size
            chunks = []
            for md_doc in md_header_splits:
                sub_chunks = self.text_splitter.create_documents([md_doc.page_content])
                for i, chunk in enumerate(sub_chunks):
                    chunk_metadata = {
                        **(metadata or {}),
                        **md_doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(sub_chunks)
                    }
                    chunks.append({
                        "text": chunk.page_content,
                        "metadata": chunk_metadata
                    })
            return chunks
        else:
            # Standard recursive splitting (legacy)
            sub_texts = self.text_splitter.split_text(text)
            chunks = [
                {
                    "text": sub_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_id": i,
                        "total_chunks": len(sub_texts),
                        "chunk_size": len(sub_text),
                    }
                }
                for i, sub_text in enumerate(sub_texts)
            ]
            return chunks

        """
        Split text into chunks.
        
        Args:
            text: The text to split.
            metadata: Optional metadata to preserve with chunks.
            
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
                    chunk_metadata = {
                        **(metadata or {}),
                        **md_doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(sub_chunks)
                    }
                    chunks.append({
                        "text": chunk.page_content,
                        "metadata": chunk_metadata
                    })
            return chunks
        else:
            # Standard recursive splitting
            sub_texts = self.text_splitter.split_text(text)
            chunks = [
                {
                    "text": sub_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_id": i,
                        "total_chunks": len(sub_texts),
                        "chunk_size": len(sub_text),
                    }
                }
                for i, sub_text in enumerate(sub_texts)
            ]
            return chunks
    
    def extract_tables_and_images(self, text: str) -> List[Dict[str, Any]]:
        """
        Placeholder for extracting tables/images using OCR/layout parsing.
        Returns a list of dicts with 'type': 'table' or 'image', and extracted content.
        """
        # TODO: Integrate OCR/table extraction libraries (e.g., camelot, pytesseract)
        return []

    def chunk_documents(self, documents: List[Dict[str, Any]], enrich: bool = True, chunk_strategy: Optional[str] = None) -> List[Dict[str, Any]]:
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
            
            chunks = self.chunk_text(text, metadata)
            results.extend(chunks)
        
        return results
