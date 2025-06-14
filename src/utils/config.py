"""
Configuration management for the RAG application.
"""
import os
from typing import Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """Configuration model for the application."""
    
    # LLM settings
    llm: Dict[str, Any] = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # PDF processing settings
    pdf: Dict[str, Any] = {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    # Retriever settings
    retriever: Dict[str, Any] = {
        "top_k": 5
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/uploads", exist_ok=True)

# Global config instance
config = Config()
