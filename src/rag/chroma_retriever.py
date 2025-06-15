"""RAG retriever implementation using ChromaDB."""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import logging
import sqlite3
import tempfile
import shutil

logger = logging.getLogger(__name__)

def get_sqlite_version():
    """Get the current SQLite version."""
    return sqlite3.sqlite_version

def is_sqlite_compatible():
    """Check if SQLite version is compatible with ChromaDB."""
    version = get_sqlite_version()
    major, minor, _ = map(int, version.split('.'))
    return (major > 3) or (major == 3 and minor >= 35)

class ChromaRetriever:
    """Retriever for RAG system using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents"):
        """Initialize the retriever with ChromaDB."""
        # Check SQLite version
        if not is_sqlite_compatible():
            logger.warning(f"SQLite version {get_sqlite_version()} is below ChromaDB's requirement (>=3.35)")
            # Create a temporary directory for ChromaDB
            self.temp_dir = tempfile.mkdtemp()
            persist_dir = self.temp_dir
        else:
            persist_dir = "chroma_db"
            self.temp_dir = None

        # Initialize ChromaDB client with appropriate settings
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize document counter
        self.doc_count = 0
        
        logger.info(f"Initialized ChromaDB with SQLite version {get_sqlite_version()}")
    
    def __del__(self):
        """Cleanup temporary directory if it exists."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retriever."""
        # Prepare documents for ChromaDB
        ids = [f"doc_{self.doc_count + i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        # Update document counter
        self.doc_count += len(documents)
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with their similarity scores
        """
        # Query the collection with a lower similarity threshold
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Log retrieval results
        logger.info(f"Retrieved {len(results['documents'][0])} documents for query: {query}")
        
        # Format results
        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            doc = {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
            }
            retrieved_docs.append(doc)
            logger.debug(f"Retrieved document {i+1} with similarity score: {doc['similarity_score']:.4f}")
        
        # Sort by similarity score
        retrieved_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Group documents by source
        source_groups = {}
        for doc in retrieved_docs:
            source = doc['metadata'].get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Reorganize documents to ensure we have complete context from each source
        reorganized_docs = []
        for source, docs in source_groups.items():
            # Sort by chunk_id to maintain order
            docs.sort(key=lambda x: x['metadata'].get('chunk_id', 0))
            reorganized_docs.extend(docs)
        
        return reorganized_docs
    
    def clear(self):
        """Clear all documents from the collection."""
        self.collection.delete(where={})
        self.doc_count = 0
        logger.info("Cleared all documents from ChromaDB") 