"""Cache utilities for the Agentic RAG application."""
import os
import json
import hashlib
from typing import Dict, Any, List, Optional
import time


class FileCache:
    """Cache system for indexed files to avoid re-processing."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the file cache.
        
        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_index_path = os.path.join(self.cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk."""
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache index: {str(e)}")
        return {}
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate a hash for a file based on its content and last modified time.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Hash string representing the file content and metadata.
        """
        try:
            file_stats = os.stat(file_path)
            modified_time = file_stats.st_mtime
            file_size = file_stats.st_size
            
            # Use only the first 64KB of large files for hashing
            with open(file_path, 'rb') as f:
                content_sample = f.read(64 * 1024)
            
            # Create hash from content sample, file size and modified time
            hasher = hashlib.sha256()
            hasher.update(content_sample)
            hasher.update(str(file_size).encode())
            hasher.update(str(modified_time).encode())
            
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash for {file_path}: {str(e)}")
            # If we can't calculate hash, use a timestamp to ensure it's processed
            return f"error_{time.time()}"
    
    def is_file_cached(self, file_path: str, chunk_size: int, chunk_overlap: int) -> bool:
        """
        Check if a file is already cached with the same processing parameters.
        
        Args:
            file_path: Path to the file.
            chunk_size: Chunk size used for processing.
            chunk_overlap: Chunk overlap used for processing.
            
        Returns:
            True if file is cached, False otherwise.
        """
        if not os.path.exists(file_path):
            return False
        
        file_hash = self._get_file_hash(file_path)
        cache_key = self._get_cache_key(file_path)
        
        if cache_key in self.cache_index:
            cache_entry = self.cache_index[cache_key]
            # Check if hash matches and parameters are the same
            return (cache_entry.get("file_hash") == file_hash and
                    cache_entry.get("chunk_size") == chunk_size and
                    cache_entry.get("chunk_overlap") == chunk_overlap)
        
        return False
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate a cache key from a file path."""
        return os.path.abspath(file_path).replace("\\", "/")
    
    def add_file_to_cache(self, file_path: str, chunk_size: int, chunk_overlap: int, 
                          document_ids: List[str]) -> None:
        """
        Add a processed file to the cache.
        
        Args:
            file_path: Path to the file.
            chunk_size: Chunk size used for processing.
            chunk_overlap: Chunk overlap used for processing.
            document_ids: IDs of the documents stored in the vector database.
        """
        if not os.path.exists(file_path):
            return
        
        file_hash = self._get_file_hash(file_path)
        cache_key = self._get_cache_key(file_path)
        
        self.cache_index[cache_key] = {
            "file_hash": file_hash,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_ids": document_ids,
            "filename": os.path.basename(file_path),
            "timestamp": time.time()
        }
        
        self._save_cache_index()
    
    def get_document_ids(self, file_path: str) -> Optional[List[str]]:
        """
        Get document IDs for a cached file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of document IDs if file is cached, None otherwise.
        """
        cache_key = self._get_cache_key(file_path)
        if cache_key in self.cache_index:
            return self.cache_index[cache_key].get("document_ids")
        return None
    
    def remove_file_from_cache(self, file_path: str) -> None:
        """
        Remove a file from the cache.
        
        Args:
            file_path: Path to the file.
        """
        cache_key = self._get_cache_key(file_path)
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def get_cached_files(self) -> List[Dict[str, Any]]:
        """
        Get a list of all cached files with their metadata.
        
        Returns:
            List of dictionaries with cached file metadata.
        """
        return [
            {
                "path": cache_key,
                "filename": entry.get("filename", os.path.basename(cache_key)),
                "timestamp": entry.get("timestamp", 0),
                "chunk_size": entry.get("chunk_size", 0),
                "chunk_overlap": entry.get("chunk_overlap", 0),
                "document_count": len(entry.get("document_ids", []))
            }
            for cache_key, entry in self.cache_index.items()
        ]
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.cache_index = {}
        self._save_cache_index()
