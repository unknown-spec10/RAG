"""TF-IDF vector store implementation for semantic search."""
from typing import List, Dict, Any, Optional
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFVectorStore:
    """TF-IDF vector store for semantic search."""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        max_features: int = 10000
    ):
        """
        Initialize the TF-IDF vector store.
        
        Args:
            index_path: Path to save/load the vector store
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        self.index_path = index_path
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.documents = []
        self.vectors = None
        
    def _initialize_vector_store(self) -> None:
        """Initialize or load the vector store."""
        if self.index_path and os.path.exists(self.index_path):
            # Load existing index
            data = np.load(self.index_path, allow_pickle=True)
            self.documents = data['documents'].tolist()
            self.vectors = data['vectors']
            self.vectorizer.fit([doc['text'] for doc in self.documents])
        else:
            # Create new empty index
            self.documents = []
            self.vectors = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        # Add new documents
        self.documents.extend(documents)
        
        # Update TF-IDF vectors
        texts = [doc['text'] for doc in self.documents]
        self.vectors = self.vectorizer.fit_transform(texts)
        
        # Save the index if path is provided
        if self.index_path:
            np.savez(
                self.index_path,
                documents=self.documents,
                vectors=self.vectors
            )
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant documents with their scores
        """
        if not self.documents or self.vectors is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= score_threshold:
                results.append({
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx].get('metadata', {}),
                    'score': score
                })
        
        return results
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.documents = []
        self.vectors = None
        
        if self.index_path and os.path.exists(self.index_path):
            os.remove(self.index_path) 