"""Simple embedding model module for text vectors."""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import hashlib
import re

class EmbeddingModel:
    """Lightweight embedding model using simple text hashing."""
    
    def __init__(
        self, 
        model_name: str = "lightweight",
        device: Optional[str] = None
    ):
        """Initialize the lightweight embedding model."""
        print("✅ Using lightweight embedding model for compatibility with Streamlit Cloud")
        self.model_name = "lightweight-hash-embeddings"
        self.device = "cpu"  # No GPU required
        self.embedding_dim = 384  # Standard dimension for compatibility
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding from text using hashing."""
        # Normalize text: lowercase, remove punctuation, extra spaces
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Get word frequency features
        words = text.split()
        total_words = len(words)
        
        # Create a deterministic hash from the text
        hash_base = hashlib.md5(text.encode()).digest()
        
        # Convert hash to a list of floats to form base embedding vector
        float_array = np.frombuffer(hash_base, dtype=np.uint8).astype(np.float32)
        
        # Expand to required dimension with some basic text features
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Fill in the embedding vector with hash values (repeated as needed)
        for i in range(self.embedding_dim):
            embedding[i] = float_array[i % len(float_array)] / 255.0  # Normalize to [0,1]
        
        # Add some simple text statistics to make embeddings more meaningful
        if total_words > 0:
            # First few dimensions hold some text statistics
            embedding[0] = len(text) / 1000.0  # Document length (normalized)
            embedding[1] = total_words / 200.0  # Word count (normalized)
            embedding[2] = len(set(words)) / total_words if total_words else 0  # Lexical diversity
            
            # Create simple n-gram features
            for i, word in enumerate(words[:20]):  # Use first 20 words only
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                position = 10 + (word_hash % (self.embedding_dim - 10))  # Start after the statistical features
                embedding[position] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
            
        return embedding
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self._generate_embedding(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            embeddings[i] = self._generate_embedding(text)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        return self._generate_embedding(query)
    
    def embed_documents(self, documents: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """Generate embeddings for a list of document chunks."""
        return [{
            **doc,
            "embedding": self._generate_embedding(doc.get(text_key, ""))
        } for doc in documents]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": 512  # Default reasonable value
        }
