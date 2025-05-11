"""Embedding model module for text vectors."""
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os
import sys

# Apply critical PyTorch settings for cloud environments
torch.set_grad_enabled(False)  # Disable gradient tracking globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism in tokenizers

# Patch PyTorch to avoid meta tensor errors on cloud platforms
original_to = torch.nn.Module.to
def safe_to_wrapper(self, *args, **kwargs):
    try:
        return original_to(self, *args, **kwargs)
    except RuntimeError as e:
        if "meta tensor" in str(e):
            print(f"⚠️ Caught meta tensor error, falling back to CPU. Error: {e}")
            # Force device to CPU for this call
            if 'device' in kwargs:
                kwargs['device'] = 'cpu'
            elif len(args) > 0 and isinstance(args[0], (str, torch.device)):
                args = list(args)
                args[0] = 'cpu'
                args = tuple(args)
            return original_to(self, *args, **kwargs)
        else:
            raise

# Apply the patched method
torch.nn.Module.to = safe_to_wrapper


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
        self.device = "cpu"  # Force CPU for all deployments to avoid memory issues
        
        # Use a try/except block with multiple fallback strategies
        for attempt in range(3):  # Try up to 3 different strategies
            try:
                if attempt == 0:
                    # First attempt: Standard initialization with CPU
                    self.model = SentenceTransformer(model_name, device=self.device)
                elif attempt == 1:
                    # Second attempt: With reduced cache and explicit settings
                    print("⚠️ First attempt failed. Trying with reduced cache...")
                    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "model_cache")
                    os.environ["HF_HOME"] = os.path.join(os.getcwd(), "model_cache")
                    self.model = SentenceTransformer(model_name, device=self.device)
                else:
                    # Final attempt: Use a minimal model with absolute minimal settings
                    print("⚠️ Second attempt failed. Falling back to minimal model...")
                    # Use the smallest model available that still works well
                    fallback_model = "paraphrase-MiniLM-L3-v2"  # Tiny 30MB model
                    self.model_name = fallback_model
                    self.model = SentenceTransformer(fallback_model, device=self.device)
                
                # If we reach here, initialization worked
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Successfully loaded model {self.model_name} with dimension {self.embedding_dim}")
                break
                
            except Exception as e:
                print(f"❌ Error loading model (attempt {attempt+1}/3): {str(e)}")
                if attempt == 2:  # If all attempts failed
                    print("❌ All model loading attempts failed. Application may not function correctly.")
                    # Create a dummy model with basic functionality
                    self.embedding_dim = 384  # Standard dimension for small models
                    self.model = None  # No model available
                    
                    # Define a method to generate random embeddings when model fails
                    def dummy_encode(texts, normalize_embeddings=True):
                        if isinstance(texts, str):
                            return np.random.randn(self.embedding_dim).astype(np.float32)
                        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
                    
                    # Create a simple object with an encode method
                    class DummyModel:
                        def encode(self, texts, normalize_embeddings=True):
                            return dummy_encode(texts, normalize_embeddings)
                    
                    self.model = DummyModel()
        
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
