"""
HybridRetriever: Combines vector similarity (ChromaRetriever) and keyword-based (TF-IDF) retrieval, with optional re-ranking.
"""
from typing import List, Dict, Any, Optional
from .chroma_retriever import ChromaRetriever
from .rag.retriever import RAGRetriever
import numpy as np

class HybridRetriever:
    def __init__(self, chroma_retriever: ChromaRetriever, tfidf_retriever: Optional[RAGRetriever] = None, reranker=None, alpha: float = 0.5):
        """
        Args:
            chroma_retriever: Vector retriever (semantic)
            tfidf_retriever: Keyword retriever (BM25/TF-IDF)
            reranker: Optional cross-encoder or re-ranking model
            alpha: Weight for blending (0.0 = only TF-IDF, 1.0 = only vector)
        """
        self.chroma = chroma_retriever
        self.tfidf = tfidf_retriever
        self.reranker = reranker
        self.alpha = alpha

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both retrievers."""
        # Add to ChromaDB (with metadata cleaning)
        self.chroma.add_documents(documents)
        
        # Add to TF-IDF retriever if available
        if self.tfidf and hasattr(self.tfidf, 'add_documents'):
            self.tfidf.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        try:
            # 1. Get vector results
            vector_docs = self.chroma.retrieve(query, top_k=top_k*2)
            # 2. Get keyword results
            keyword_docs = self.tfidf.retrieve(query, top_k=top_k*2) if self.tfidf else []

            # 3. Combine by unique doc id (or text+source)
            doc_map = {}
            for doc in vector_docs:
                key = (doc['metadata'].get('source'), doc['text'])
                doc_map[key] = {**doc, 'vector_score': doc.get('similarity_score', 0), 'keyword_score': 0}
            for doc in keyword_docs:
                key = (doc['metadata'].get('source'), doc['text'])
                if key in doc_map:
                    doc_map[key]['keyword_score'] = doc.get('similarity_score', 0)
                else:
                    doc_map[key] = {**doc, 'vector_score': 0, 'keyword_score': doc.get('similarity_score', 0)}

            # 4. Blend scores
            for d in doc_map.values():
                d['hybrid_score'] = self.alpha * d['vector_score'] + (1 - self.alpha) * d['keyword_score']

            # 5. Top-N by hybrid score
            docs = sorted(doc_map.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]

            # 6. Optional: Re-rank
            if self.reranker:
                try:
                    # Reranker returns sorted list of docs
                    docs = self.reranker.rerank(query, docs)
                except Exception as e:
                    print(f"Reranking failed, using hybrid scores: {e}")
                    # Continue with hybrid scores if reranking fails

            return docs
        except Exception as e:
            print(f"Hybrid retrieval failed: {e}")
            # Fallback to just vector retrieval
            try:
                return self.chroma.retrieve(query, top_k=top_k)
            except Exception as e2:
                print(f"Vector retrieval also failed: {e2}")
                return []

from .rerankers import JinaReranker

# Use JinaReranker for production re-ranking
# Example usage:
# reranker = JinaReranker(model_name="jina-reranker-v1-base-en")
# hybrid = HybridRetriever(chroma_retriever, tfidf_retriever, reranker=reranker)
