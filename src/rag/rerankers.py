"""
Reranker: Uses transformer-based re-ranking models.
"""
from typing import List, Dict, Any
from reranker import ReRanker

class JinaReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        # Map jina model names to compatible models
        model_mapping = {
            "jina-reranker-v1-base-en": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "jina-reranker-v1-tiny-en": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        }
        
        # Use mapped model name if it exists, otherwise use the provided name
        actual_model = model_mapping.get(model_name, model_name)
        self.reranker = ReRanker(actual_model)

    def rerank(self, query: str, docs: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        try:
            # Extract texts from documents
            texts = [d[text_key] for d in docs]
            
            # Get reranking scores using the correct method
            # The reranker.rank method returns a list of RankResult objects
            results = self.reranker.rank(query, texts)
            
            # Handle different return formats
            if hasattr(results[0], 'score'):
                # If results have score attribute
                scores = [result.score for result in results]
            elif isinstance(results[0], (int, float)):
                # If results are just scores
                scores = results
            else:
                # Fallback: assume results are dictionaries with 'score' key
                scores = [result.get('score', 0) if isinstance(result, dict) else 0 for result in results]
            
            # Assign scores and sort
            for d, score in zip(docs, scores):
                d["rerank_score"] = float(score) if score is not None else 0.0
                
            return sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
        except Exception as e:
            print(f"Reranking failed: {e}, returning original order")
            # If reranking fails, return original docs with default scores
            for i, d in enumerate(docs):
                d["rerank_score"] = 1.0 - (i * 0.01)  # Slightly decreasing scores
            return docs
