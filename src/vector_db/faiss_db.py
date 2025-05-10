import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
import pickle
from pathlib import Path

class FAISSVectorDB:
    """Vector database implementation using FAISS."""
    
    def __init__(
        self, 
        persist_directory: str,
        collection_name: str = "documents",
        dimension: int = 384,  # Default dimension for all-MiniLM-L6-v2
        metric_type: int = faiss.METRIC_INNER_PRODUCT
    ):
        """
        Initialize the FAISS vector database.
        
        Args:
            persist_directory: Directory to persist the database.
            collection_name: Name of the collection.
            dimension: Dimension of the embeddings.
            metric_type: Distance metric to use (faiss.METRIC_INNER_PRODUCT or faiss.METRIC_L2).
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize index
        self.index = self._load_or_create_index()
        self.metadata = self._load_or_create_metadata()
        
    def _get_index_path(self) -> str:
        """Get the path to the FAISS index file."""
        return os.path.join(self.persist_directory, f"{self.collection_name}.index")
    
    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file."""
        return os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Load existing index or create a new one."""
        index_path = self._get_index_path()
        
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        else:
            return faiss.IndexFlatIP(self.dimension)  # Using Inner Product for cosine similarity
    
    def _load_or_create_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create a new dictionary."""
        metadata_path = self._get_metadata_path()
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embedding_key: str = "embedding",
        text_key: str = "text",
        metadata_key: str = "metadata",
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with embeddings.
            embedding_key: Key for embeddings in document dictionaries.
            text_key: Key for text in document dictionaries.
            metadata_key: Key for metadata in document dictionaries.
            ids: Optional list of document IDs. If None, will generate UUIDs.
            
        Returns:
            List of document IDs.
        """
        if not documents:
            return []
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        embeddings = np.array([doc[embedding_key] for doc in documents], dtype=np.float32)
        texts = [doc[text_key] for doc in documents]
        metadatas = [doc[metadata_key] for doc in documents]
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Update metadata
        for doc_id, text, metadata in zip(ids, texts, metadatas):
            self.metadata[doc_id] = {
                "text": text,
                "metadata": metadata
            }
        
        # Save index and metadata
        self._save_index()
        self._save_metadata()
        
        return ids
    
    def _save_index(self):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, self._get_index_path())
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self._get_metadata_path(), 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def query(
        self,
        query_embedding: Union[List[float], np.ndarray],
        n_results: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database with an embedding.
        
        Args:
            query_embedding: Query embedding.
            n_results: Number of results to return.
            filter_criteria: Optional filter criteria for metadata.
            
        Returns:
            Query results.
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Normalize the query embedding
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, n_results)
        
        # Get results
        results = {
            "ids": [],
            "distances": distances[0].tolist(),
            "documents": [],
            "metadatas": []
        }
        
        for idx in indices[0]:
            if idx >= len(self.metadata):
                continue
                
            doc_id = list(self.metadata.keys())[idx]
            doc_data = self.metadata[doc_id]
            
            results["ids"].append(doc_id)
            results["documents"].append(doc_data["text"])
            results["metadatas"].append(doc_data["metadata"])
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Document dictionary if found, None otherwise.
        """
        return self.metadata.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if deleted, False otherwise.
        """
        if doc_id in self.metadata:
            del self.metadata[doc_id]
            self._save_metadata()
            return True
        return False
    
    def count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents.
        """
        return len(self.metadata)
    
    def update_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
        embedding_key: str = "embedding",
        text_key: str = "text",
        metadata_key: str = "metadata"
    ) -> bool:
        """
        Update a document in the vector database.
        
        Args:
            doc_id: Document ID.
            document: Updated document dictionary.
            embedding_key: Key for embedding in document dictionary.
            text_key: Key for text in document dictionary.
            metadata_key: Key for metadata in document dictionary.
            
        Returns:
            True if updated, False otherwise.
        """
        if doc_id not in self.metadata:
            return False
            
        self.metadata[doc_id] = {
            "text": document[text_key],
            "metadata": document[metadata_key]
        }
        
        self._save_metadata()
        return True
