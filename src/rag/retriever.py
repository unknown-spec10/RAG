"""RAG retriever implementation using TF-IDF based similarity."""
from typing import List, Dict, Any
import numpy as np
from .embeddings import TFIDFEmbeddings

class RAGRetriever:
    """Retriever for RAG system using TF-IDF based similarity."""
    
    def __init__(self, embeddings: TFIDFEmbeddings = None):
        """Initialize the retriever."""
        self.embeddings = embeddings or TFIDFEmbeddings()
        self.documents = []
        self.document_embeddings = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        self.documents.extend(documents)
        # Generate embeddings for new documents
        new_embeddings = self.embeddings.embed_documents(documents)
        self.document_embeddings.extend(new_embeddings)
    
    def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Compute cosine similarity between query and document embeddings."""
        # Normalize the vectors
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(doc_embedding)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
            
        # Compute cosine similarity
        return np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
    
    def retrieve(self, query: str, documents: List[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            documents: Optional list of documents to search in. If None, uses stored documents.
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with their similarity scores
        """
        # Use provided documents or stored documents
        docs_to_search = documents if documents is not None else self.documents
        embeddings_to_search = self.embeddings.embed_documents(docs_to_search)
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Compute similarities
        similarities = [
            self._compute_similarity(query_embedding, doc_embedding)
            for doc_embedding in embeddings_to_search
        ]
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return documents with their similarity scores
        return [
            {
                **docs_to_search[i],
                "similarity_score": float(similarities[i])
            }
            for i in top_indices
        ]
