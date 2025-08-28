"""
Web Search Agent using SERP API for real-time search.
Handles web search with privacy and security considerations.
"""
from typing import List, Dict, Any, Optional
import logging
import os
from serpapi import GoogleSearch
import streamlit as st

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Web search agent that uses SERP API for real-time search.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search agent.
        
        Args:
            api_key: SERP API key (optional, can be read from secrets)
        """
        self.api_key = api_key or self._get_serp_api_key()
        self.max_results = 5  # Limit results to avoid too much external data
        
    def _get_serp_api_key(self) -> Optional[str]:
        """Get SERP API key from secrets or environment."""
        try:
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and 'serp_api_key' in st.secrets:
                return st.secrets['serp_api_key']
        except Exception:
            pass
        
        # Try environment variable
        return os.getenv('SERP_API_KEY')

    def search(self, query: str, search_type: str = "search") -> List[Dict[str, Any]]:
        """
        Perform web search using SERP API.
        
        Args:
            query: Search query (should be filtered/sanitized)
            search_type: Type of search ("search", "news", "scholar")
            
        Returns:
            List of search results
        """
        if not self.api_key:
            logger.warning("SERP API key not found. Web search unavailable.")
            return []
        
        if not query or len(query.strip()) < 3:
            logger.warning("Query too short for web search")
            return []
        
        try:
            # Configure search parameters
            search_params = {
                "q": query,
                "api_key": self.api_key,
                "num": self.max_results,
                "safe": "active",  # Enable safe search
            }
            
            # Add search type specific parameters
            if search_type == "news":
                search_params["tbm"] = "nws"
                search_params["tbs"] = "qdr:m"  # Last month for news
            elif search_type == "scholar":
                search_params["engine"] = "google_scholar"
            
            # Perform search
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            # Process and format results
            return self._process_search_results(results, search_type)
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []

    def _process_search_results(self, results: Dict[str, Any], search_type: str) -> List[Dict[str, Any]]:
        """Process and format search results."""
        processed_results = []
        
        try:
            # Handle different result types
            if search_type == "news":
                result_key = "news_results"
            elif search_type == "scholar":
                result_key = "organic_results"
            else:
                result_key = "organic_results"
            
            raw_results = results.get(result_key, [])
            
            for i, result in enumerate(raw_results[:self.max_results]):
                processed_result = {
                    "rank": i + 1,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", result.get("summary", "")),
                    "link": result.get("link", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", ""),
                    "search_type": search_type
                }
                
                # Add additional fields for news
                if search_type == "news":
                    processed_result.update({
                        "published_date": result.get("date", ""),
                        "thumbnail": result.get("thumbnail", "")
                    })
                
                # Add additional fields for scholar
                elif search_type == "scholar":
                    processed_result.update({
                        "cited_by": result.get("cited_by", {}).get("value", 0),
                        "publication_info": result.get("publication_info", {})
                    })
                
                processed_results.append(processed_result)
                
        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
        
        return processed_results

    def search_news(self, query: str) -> List[Dict[str, Any]]:
        """Search for recent news articles."""
        return self.search(query, "news")

    def search_scholarly(self, query: str) -> List[Dict[str, Any]]:
        """Search for scholarly articles."""
        return self.search(query, "scholar")

    def search_general(self, query: str) -> List[Dict[str, Any]]:
        """General web search."""
        return self.search(query, "search")

    def is_available(self) -> bool:
        """Check if web search is available (API key configured)."""
        return self.api_key is not None

    def format_results_for_context(self, results: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """
        Format search results for inclusion in LLM context.
        
        Args:
            results: Search results
            max_length: Maximum character length for formatted results
            
        Returns:
            Formatted string suitable for LLM context
        """
        if not results:
            return "No web search results found."
        
        formatted = "Web Search Results:\n\n"
        current_length = len(formatted)
        
        for result in results:
            result_text = f"{result['rank']}. {result['title']}\n"
            result_text += f"   {result['snippet']}\n"
            result_text += f"   Source: {result['source']}\n"
            if result.get('date'):
                result_text += f"   Date: {result['date']}\n"
            result_text += "\n"
            
            # Check if adding this result would exceed max_length
            if current_length + len(result_text) > max_length:
                formatted += "... (additional results truncated)\n"
                break
            
            formatted += result_text
            current_length += len(result_text)
        
        return formatted

    def get_search_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of search results.
        
        Args:
            results: Search results
            
        Returns:
            Summary information
        """
        if not results:
            return {
                "total_results": 0,
                "search_types": [],
                "sources": [],
                "date_range": None
            }
        
        search_types = list(set(result.get('search_type', 'search') for result in results))
        sources = list(set(result.get('source', 'Unknown') for result in results if result.get('source')))
        dates = [result.get('date') for result in results if result.get('date')]
        
        return {
            "total_results": len(results),
            "search_types": search_types,
            "sources": sources[:10],  # Limit to 10 sources
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            }
        }
