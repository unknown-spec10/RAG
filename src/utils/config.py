"""Configuration utilities for the Agentic RAG application."""
import os
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the Agentic RAG application."""
    
    DEFAULT_CONFIG = {
        "pdf_processor": {
            "use_pdfplumber": False,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "storage_type": "cloud",  # Can be 'cloud' or 'local'
            "cloud_storage_provider": "s3",  # Can be 's3' or 'azure'
            "bucket_name": "agentic-rag-documents",
            "region": "us-east-1"
        },
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",
            "device": None,  # Auto-detect
        },
        "vector_db": {
            "storage_type": "local",  # Can be 'local' or 'memory'
            "persist_directory": "data/faiss_db",
            "collection_name": "documents",
            "dimension": 384,  # Default dimension for all-MiniLM-L6-v2
            "metric_type": "inner_product",  # or 'l2'
        },
        "llm": {
            "model_name": "llama3",
            "provider": "ollama",  # Can be 'ollama' or 'groq'
            "groq_api_key": None,  # Groq API key if using Groq
            "groq_model": "llama2-70b-chat",  # Default Groq model
            "temperature": 0.1,
            "context_window": 8192,
            "max_tokens": 1024,
        },
        "retriever": {
            "top_k": 5,
            "rerank": True,
        },
        "app": {
            "data_dir": "data",
            "pdf_dir": "data/pdfs",
            "debug": False,
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to the config file. If None, use default config.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                
                # Update config with file values
                self._update_nested_dict(self.config, file_config)
            except Exception as e:
                print(f"Error loading config from {config_path}: {str(e)}")
        
        # Override with environment variables
        self._load_from_env()
        
        # Create required directories
        self._create_directories()
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update a nested dictionary with another dictionary.
        
        Args:
            d: Dictionary to update.
            u: Dictionary with updates.
            
        Returns:
            Updated dictionary.
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Example: AGENTIC_RAG_LLM_MODEL_NAME -> config["llm"]["model_name"]
        prefix = "AGENTIC_RAG_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split("_")
                
                if len(parts) >= 2:
                    section = parts[0]
                    param = "_".join(parts[1:])
                    
                    if section in self.config and param in self.config[section]:
                        # Convert value to appropriate type
                        current_value = self.config[section][param]
                        
                        if isinstance(current_value, bool):
                            self.config[section][param] = value.lower() in ("true", "1", "yes")
                        elif isinstance(current_value, int):
                            try:
                                self.config[section][param] = int(value)
                            except ValueError:
                                pass
                        elif isinstance(current_value, float):
                            try:
                                self.config[section][param] = float(value)
                            except ValueError:
                                pass
                        else:
                            self.config[section][param] = value
    
    def _create_directories(self):
        """Create required directories."""
        os.makedirs(self.config["app"]["data_dir"], exist_ok=True)
        os.makedirs(self.config["app"]["pdf_dir"], exist_ok=True)
        os.makedirs(self.config["vector_db"]["persist_directory"], exist_ok=True)
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section.
            key: Configuration key within section. If None, return the entire section.
            
        Returns:
            Configuration value.
        """
        if section not in self.config:
            return None
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key)
    
    def save(self, config_path: str):
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration.
        """
        try:
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {config_path}: {str(e)}")
