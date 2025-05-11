"""ChromaDB vector database module for storing and retrieving document embeddings."""
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import uuid


class ChromaVectorDB:
    """Vector database implementation using ChromaDB."""
    
    def __init__(
        self, 
        persist_directory: str,
        collection_name: str = "documents",
        distance_func: str = "cosine",
        storage_type: str = "local",
        cloud_storage_provider: str = "s3",
        bucket_name: str = "agentic-rag-chroma",
        region: str = "us-east-1"
    ):
        """
        Initialize the Chroma vector database.
        
        Args:
            persist_directory: Directory to persist the database (for local storage).
            collection_name: Name of the collection.
            distance_func: Distance function to use (cosine, dot, l2).
            storage_type: Type of storage ('local' or 'cloud').
            cloud_storage_provider: Cloud storage provider ('s3' or 'azure').
            bucket_name: Name of the cloud storage bucket.
            region: Cloud region for storage.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_func = distance_func
        self.storage_type = storage_type
        self.cloud_storage_provider = cloud_storage_provider
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize ChromaDB
        if storage_type == "local":
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # For cloud storage, we'll use a different client
            self.client = self._initialize_cloud_client()
        
        self.collection = self._get_or_create_collection()
    
    def _initialize_cloud_client(self):
        """Initialize the cloud storage client."""
        if self.cloud_storage_provider == "s3":
            import boto3
            from chromadb.config import Settings
            
            # Initialize S3 client
            s3_client = boto3.client('s3', region_name=self.region)
            
            # Create bucket if it doesn't exist
            try:
                s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            except s3_client.exceptions.BucketAlreadyExists:
                pass
            
            # Initialize Chroma with S3 settings
            return chromadb.HttpClient(
                settings=Settings(
                    chroma_db_impl="rest",
                    host="chroma-server",  # This would be your Chroma server endpoint
                    port=8000,
                    allow_reset=True
                )
            )
        
        # List all collections to check if our collection exists
        all_collections = self.client.list_collections()
        collection_exists = any(col.name == collection_name for col in all_collections)
        
        if collection_exists:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll handle embeddings externally
            )
        else:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,  # We'll handle embeddings externally
                metadata={"hnsw:space": distance_func}
            )
    
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
        
        embeddings = [doc.get(embedding_key, []) for doc in documents]
        texts = [doc.get(text_key, "") for doc in documents]
        metadatas = [doc.get(metadata_key, {}) for doc in documents]
        
        # Add documents to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
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
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_criteria
        )
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Document dictionary if found, None otherwise.
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["embeddings", "documents", "metadatas"])
            
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "embedding": result["embeddings"][0],
                    "metadata": result["metadatas"][0]
                }
            return None
        except Exception:
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if deleted, False otherwise.
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
            
    def count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents.
        """
        return self.collection.count()
    
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
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[document.get(embedding_key, [])],
                documents=[document.get(text_key, "")],
                metadatas=[document.get(metadata_key, {})]
            )
            return True
        except Exception:
            return False
