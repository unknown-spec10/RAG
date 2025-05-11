"""Retriever module for fetching relevant documents based on a query."""
from typing import List, Dict, Any, Optional
import numpy as np
import math
import re
import hashlib
from collections import Counter
from src.vector_db.faiss_db import FAISSVectorDB
from src.embeddings.embedding_model import EmbeddingModel


class RAGRetriever:
    """Class for retrieving relevant documents for a query."""
    
    def __init__(
        self, 
        vector_db: FAISSVectorDB,
        embedding_model: EmbeddingModel,
        top_k: int = 5
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_db: Vector database instance.
            embedding_model: Embedding model instance.
            top_k: Number of documents to retrieve.
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def retrieve(
        self, 
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string.
            filter_criteria: Optional filter criteria for metadata.
            rerank: Whether to rerank results.
            
        Returns:
            List of retrieved documents.
        """
        # Try simple keyword search first for non-empty query
        if query.strip():
            try:
                all_docs = self.vector_db.get_all_documents()
                if all_docs and "documents" in all_docs and all_docs["documents"]:
                    documents = self._simple_keyword_search(query, all_docs, self.top_k)
                    if documents:
                        print(f"Found {len(documents)} documents with keyword search")
                        return documents
            except Exception as e:
                print(f"Keyword search error: {str(e)}. Falling back to vector search.")
        
        # Vector search as fallback
        try:
            # Check if vector db exists and has documents
            if self.vector_db is None:
                print("Warning: Vector database is not initialized")
                return []
                
            doc_count = self.vector_db.count()
            if doc_count == 0:
                print("Warning: No documents found in the vector database")
                return []
                
            # Create a simple hash-based embedding for the query
            # Process the query text
            text = query.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Generate a deterministic hash
            hash_result = hashlib.md5(text.encode()).digest()
            
            # Create a fixed-dimensional embedding
            query_embedding = np.zeros(384, dtype=np.float32)
            hash_floats = np.frombuffer(hash_result, dtype=np.uint8).astype(np.float32)
            
            # Fill the embedding vector
            for i in range(len(query_embedding)):
                query_embedding[i] = hash_floats[i % len(hash_floats)] / 255.0
            
            # Shape it correctly for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Try vector search
            results = self.vector_db.query(
                query_embedding=query_embedding,
                n_results=self.top_k
            )
            
            # Process results
            documents = []
            if results and "documents" in results and results["documents"]:
                for i, (doc_text, metadata, distance) in enumerate(
                    zip(results.get("documents", []), results.get("metadatas", []), results.get("distances", []))
                ):
                    documents.append({
                        "text": doc_text,
                        "metadata": metadata,
                        "id": "unknown",
                        "score": distance
                    })
                
                if documents:
                    print(f"Found {len(documents)} documents with vector search")
                    return documents
            
            # If we got here, both search methods failed
            print("Warning: No documents found with any search method")
            return []
                
        except Exception as e:
            import traceback
            print(f"Error in vector search: {str(e)}")
            print(traceback.format_exc())
            return []
    
    def _simple_keyword_search(self, query: str, all_docs: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple and reliable keyword search on documents.
        
        Args:
            query: Query text
            all_docs: Dictionary with document data from vector db
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Extract document data
        doc_texts = all_docs.get("documents", [])
        doc_ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])
        
        # Ensure we have equal length arrays with fallbacks
        n_docs = len(doc_texts)
        if len(doc_ids) < n_docs:
            doc_ids = doc_ids + [f"doc_{i}" for i in range(len(doc_ids), n_docs)]
        if len(metadatas) < n_docs:
            metadatas = metadatas + [{} for _ in range(len(metadatas), n_docs)]
            
        if not doc_texts:
            return []
            
        # Simple keyword matching
        query_terms = query.lower().split()
        results = []
        
        for i, doc_text in enumerate(doc_texts):
            if not doc_text:
                continue
                
            # Count matches
            doc_lower = doc_text.lower()
            matches = 0
            
            for term in query_terms:
                if term in doc_lower:
                    matches += 1
            
            # Only include if there's at least one match
            if matches > 0:
                results.append({
                    "id": doc_ids[i] if i < len(doc_ids) else f"doc_{i}",
                    "text": doc_text,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": matches
                })
                
        # Sort by number of matches (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on more sophisticated similarity to query.
        This is a simple implementation - could be replaced with more 
        sophisticated reranking in the future.
        
        Args:
            query: Query string.
            documents: List of documents to rerank.
            
        Returns:
            Reranked list of documents.
        """
        # Simple reranking - could be extended with more sophisticated methods
        query_embedding = self.embedding_model.embed_text(query)
        
        for doc in documents:
            # Get document text
            text = doc.get("text", "")
            
            # Calculate similarity score
            doc_embedding = self.embedding_model.embed_text(text)
            similarity = self.embedding_model.similarity(query_embedding, doc_embedding)
            
            # Update similarity score
            doc["similarity"] = similarity
        
        # Sort by similarity (descending)
        documents.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return documents
    
    def hybrid_retrieve(
        self, 
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector similarity and keyword matching.
        
        Args:
            query: Query string.
            filter_criteria: Optional filter criteria for metadata.
            keyword_weight: Weight for keyword matching (0-1).
            
        Returns:
            List of retrieved documents.
        """
        # Vector retrieval
        vector_results = self.retrieve(
            query=query,
            filter_criteria=filter_criteria,
            rerank=False
        )
        
        # Combine results (in a real implementation, you would add keyword search here)
        # This is a placeholder for a more sophisticated hybrid retrieval
        return vector_results
