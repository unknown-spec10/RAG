"""TF-IDF based embeddings implementation."""
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFEmbeddings:
    """TF-IDF based embeddings for document retrieval."""
    
    def __init__(self):
        """Initialize the TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
    
    def _fit_if_needed(self, texts: List[str] = None):
        """Fit the vectorizer if it hasn't been fitted yet."""
        if not self.is_fitted and texts:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if non_empty_texts:
                self.vectorizer.fit(non_empty_texts)
                self.is_fitted = True
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a query text."""
        if not text.strip():
            # Return zero vector for empty queries
            return np.zeros(self.vectorizer.get_feature_names_out().shape[0]) if self.is_fitted else np.zeros(1000)
            
        if not self.is_fitted:
            # If not fitted, fit with just this text
            self._fit_if_needed([text])
        return self.vectorizer.transform([text]).toarray()[0]
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for a list of documents."""
        texts = [doc["text"] for doc in documents]
        self._fit_if_needed(texts)
        
        # Handle empty documents
        if not self.is_fitted:
            # If no non-empty documents, return zero vectors
            return [np.zeros(1000) for _ in documents]
            
        # Transform all documents, including empty ones
        embeddings = []
        for text in texts:
            if not text.strip():
                # Return zero vector for empty documents
                embeddings.append(np.zeros(self.vectorizer.get_feature_names_out().shape[0]))
            else:
                embeddings.append(self.vectorizer.transform([text]).toarray()[0])
        return embeddings
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary) of the vectorizer."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist() 