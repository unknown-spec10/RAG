"""Retriever module for fetching relevant documents based on a query."""
from typing import List, Dict, Any, Optional
import numpy as np
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
        try:
            # Check if vector db is initialized properly
            if self.vector_db is None:
                print("Error: Vector database is not initialized")
                return []
                
            # Check if there are any documents in the database
            doc_count = self.vector_db.count()
            if doc_count == 0:
                print("Warning: No documents found in the vector database")
                return []
                
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_text(query)
            
            # Query the vector database
            results = self.vector_db.query(
                query_embedding=query_embedding,
                n_results=min(self.top_k, doc_count),  # Make sure not to request more docs than exist
                filter_criteria=filter_criteria
            )
            
            # Make sure results contain the expected keys
            if "documents" not in results or not results["documents"]:
                print(f"Warning: No documents returned in query results. Keys: {list(results.keys())}")
                return []
            
            # Extract documents
            documents = []
            for i, (doc_text, metadata, doc_id, distance) in enumerate(zip(
                results.get("documents", []),
                results.get("metadatas", [{}] * len(results.get("documents", []))),
                results.get("ids", ["unknown"] * len(results.get("documents", []))),
                results.get("distances", [0.0] * len(results.get("documents", [])))
            )):
                documents.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "id": doc_id,
                    "score": distance
                })
                
            # Optional reranking
            if rerank and documents:
                documents = self._rerank_documents(query, documents)
                
            # Add debug information about retrieved documents
            print(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
        except Exception as e:
            import traceback
            print(f"Error retrieving documents: {str(e)}")
            print(traceback.format_exc())
            return []
    
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
