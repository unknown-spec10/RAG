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
            # Initialize with empty lists for parallel arrays
            return {
                "ids": [],
                "texts": [],
                "metadatas": []
            }
    
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
        
        # Update metadata - store in parallel arrays
        if "ids" not in self.metadata:
            self.metadata["ids"] = []
            self.metadata["texts"] = []
            self.metadata["metadatas"] = []
            
        # Add new documents
        self.metadata["ids"].extend(ids)
        self.metadata["texts"].extend(texts)
        self.metadata["metadatas"].extend(metadatas)
        
        # Also keep the old format for backward compatibility
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
        # Debug the input
        print(f"Query embedding input type: {type(query_embedding)}")
        
        if query_embedding is None:
            print("Error: Query embedding is None")
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
        
        # Ensure query_embedding is numpy array
        if not isinstance(query_embedding, np.ndarray):
            try:
                query_embedding = np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                print(f"Error converting query embedding to numpy array: {str(e)}")
                return {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": []
                }
        
        # Flatten if it's a 2D embedding with shape (1, D)
        if len(query_embedding.shape) > 1 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]  # Extract the vector
            print(f"Flattened query embedding shape: {query_embedding.shape}")
        
        # Ensure it's 1D before FAISS search
        if len(query_embedding.shape) > 1:
            print(f"Warning: Reshaping multi-dimensional embedding: {query_embedding.shape}")
            query_embedding = query_embedding.reshape(-1)  # Flatten to 1D
        
        # Make sure the dimensions match expected size (384)
        if query_embedding.shape[0] != self.dimension:
            print(f"Warning: Query embedding dimension {query_embedding.shape[0]} doesn't match index dimension {self.dimension}")
            # Create a fixed-size embedding
            fixed_embedding = np.zeros(self.dimension, dtype=np.float32)
            # Copy what we can
            min_dim = min(query_embedding.shape[0], self.dimension)
            fixed_embedding[:min_dim] = query_embedding[:min_dim]
            query_embedding = fixed_embedding
            
        # Debug information
        print(f"Final query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        
        # Reshape for FAISS (expects 2D array with shape [n_queries, dimension])
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        try:
            D, I = self.index.search(query_embedding, n_results)
            print(f"FAISS search successful. Results shape: D={D.shape}, I={I.shape}")
        except Exception as e:
            import traceback
            print(f"Error in FAISS search: {str(e)}")
            print(f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
            print(traceback.format_exc())
            # Return empty results
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
        
        # Extract results
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        # Check if we have parallel lists in metadata
        if all(key in self.metadata for key in ["ids", "texts", "metadatas"]):
            # Using the new parallel array format
            for i, (idx, distance) in enumerate(zip(I[0], D[0])):
                if idx != -1 and idx < len(self.metadata["ids"]):  # Valid index
                    doc_id = self.metadata["ids"][idx]
                    text = self.metadata["texts"][idx]
                    metadata = self.metadata["metadatas"][idx]
                    
                    # Apply filter criteria if provided
                    if filter_criteria and not self._matches_filter(metadata, filter_criteria):
                        continue
                    
                    results["ids"].append(doc_id)
                    results["documents"].append(text)
                    results["metadatas"].append(metadata)
                    results["distances"].append(float(distance))
        else:
            # Fallback to the old dictionary format
            for i, (idx, distance) in enumerate(zip(I[0], D[0])):
                if idx != -1:  # -1 means no result found
                    # Get document ID for this index - we need to search through all docs
                    found = False
                    for doc_id, info in self.metadata.items():
                        # Skip non-document entries
                        if not isinstance(doc_id, str) or not isinstance(info, dict):
                            continue
                            
                        # Apply filter criteria if provided
                        if filter_criteria and not self._matches_filter(info["metadata"], filter_criteria):
                            continue
                        
                        results["ids"].append(doc_id)
                        results["documents"].append(info["text"])
                        results["metadatas"].append(info["metadata"])
                        results["distances"].append(float(distance))
                        found = True
                        break
                        
                    if found and len(results["ids"]) >= n_results:
                        break
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Document dictionary if found, None otherwise.
        """
        if doc_id in self.metadata["ids"]:
            idx = self.metadata["ids"].index(doc_id)
            return {
                "text": self.metadata["texts"][idx],
                "metadata": self.metadata["metadatas"][idx]
            }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if deleted, False otherwise.
        """
        if doc_id in self.metadata["ids"]:
            idx = self.metadata["ids"].index(doc_id)
            del self.metadata["ids"][idx]
            del self.metadata["texts"][idx]
            del self.metadata["metadatas"][idx]
            del self.metadata[doc_id]
            self._save_metadata()
            return True
        return False
    
    def count(self) -> int:
        """Get the number of documents in the database."""
        try:
            return self.index.ntotal
        except Exception as e:
            print(f"Error counting documents: {str(e)}")
            return 0
            
    def get_all_documents(self) -> Dict[str, List]:
        """Get all documents from the database.
        
        Returns:
            Dictionary with lists of documents, metadatas, ids and dummy distances.
        """
        try:
            if not self.metadata or not self.metadata.get("texts") or not self.metadata.get("ids"):
                print("No documents found in database")
                return {}
                
            doc_ids = self.metadata.get("ids", [])
            texts = self.metadata.get("texts", [])
            metadatas = self.metadata.get("metadatas", [])
            
            # Create a dummy distances list (not used for keyword search)
            distances = [1.0] * len(doc_ids)
            
            return {
                "ids": doc_ids,
                "documents": texts,
                "metadatas": metadatas,
                "distances": distances
            }
        except Exception as e:
            import traceback
            print(f"Error getting all documents: {str(e)}")
            print(traceback.format_exc())
            return {}
    
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
