"""Embedding model module for text vectors."""
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import asyncio


class EmbeddingModel:
    """Class for generating embeddings from text using sentence transformers."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                Options include: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 
                'bge-small-en-v1.5', 'bge-base-en-v1.5'
            device: Device to use for inference ('cpu', 'cuda', or None for auto).
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)
    
    def embed_documents(self, documents: List[Dict[str, Any]], 
                        text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            documents: List of document dictionaries, each with a text field.
            text_key: Key to access the text in each document.
            
        Returns:
            List of document dictionaries with added embeddings.
        """
        texts = [doc.get(text_key, "") for doc in documents]
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to documents
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            documents[i] = {
                **doc,
                "embedding": embedding
            }
            
        return documents
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": self.model.get_max_seq_length()
        }
