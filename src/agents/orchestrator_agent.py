"""
Orchestrator Agent that decides whether to use local DB or web search.
Uses LLM-based decision making for dynamic and intelligent query analysis.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
import logging
import json
from datetime import datetime
from enum import Enum
from langchain_groq import ChatGroq
import os

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the orchestrator can identify."""
    LOCAL_DOCUMENT = "local_document"  # Query about uploaded documents
    CURRENT_EVENTS = "current_events"  # Recent news, updates
    GENERAL_KNOWLEDGE = "general_knowledge"  # Public information
    TECHNICAL_UPDATES = "technical_updates"  # Latest versions, releases
    MIXED = "mixed"  # Requires both local and web search

class PrivacyLevel(Enum):
    """Privacy levels for query classification."""
    SAFE = "safe"  # Safe to send to external APIs
    SENSITIVE = "sensitive"  # Contains sensitive info, needs filtering
    CONFIDENTIAL = "confidential"  # Should not be sent externally

class OrchestratorAgent:
    """
    LLM-powered orchestrator agent that dynamically decides search strategy and ensures privacy.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama3-70b-8192"):
        """
        Initialize the orchestrator with LLM capabilities.
        
        Args:
            api_key: Groq API key for LLM
            model_name: LLM model to use for decision making
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        
        # Initialize LLM for decision making
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=model_name,
                    temperature=0.1  # Low temperature for consistent decisions
                )
                self.llm_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for orchestrator: {e}")
                self.llm_available = False
        else:
            logger.warning("No API key provided for orchestrator LLM")
            self.llm_available = False

    def analyze_query(self, query: str, has_local_docs: bool = True) -> Dict[str, Any]:
        """
        Analyze query using LLM to determine search strategy and privacy level.
        
        Args:
            query: User query
            has_local_docs: Whether local documents are available
            
        Returns:
            Dictionary with analysis results
        """
        if self.llm_available:
            return self._llm_analyze_query(query, has_local_docs)
        else:
            # Fallback to rule-based analysis if LLM not available
            return self._fallback_analyze_query(query, has_local_docs)

    def _llm_analyze_query(self, query: str, has_local_docs: bool) -> Dict[str, Any]:
        """Use LLM to analyze the query and make decisions."""
        
        analysis_prompt = f"""You are an intelligent query analyzer that determines the best search strategy for a RAG system. 

Analyze this user query and provide a JSON response with your analysis:

User Query: "{query}"
Local Documents Available: {has_local_docs}

Please analyze the query and respond with a JSON object containing:

1. "privacy_level": One of ["safe", "sensitive", "confidential"]
   - "safe": Query can be safely sent to external APIs
   - "sensitive": Contains some sensitive info that should be filtered
   - "confidential": Contains confidential information, should never be sent externally

2. "query_type": One of ["local_document", "current_events", "general_knowledge", "technical_updates", "mixed"]
   - "local_document": Query about uploaded/local documents
   - "current_events": Recent news, current events, today's information
   - "general_knowledge": General public knowledge questions
   - "technical_updates": Latest versions, releases, updates of technology/software
   - "mixed": Requires both local and external information

3. "requires_web_search": boolean - whether web search would be beneficial
4. "requires_local_search": boolean - whether local documents should be searched
5. "priority": One of ["local", "web", "hybrid"] - which source to prioritize
6. "reasoning": string - explanation of your decision
7. "filtered_query": string - a sanitized version of the query safe for external APIs (remove any sensitive references)

Consider these factors:
- References to "this document", "uploaded file", "our company", "internal" suggest local focus
- Words like "latest", "current", "recent", "today", "breaking" suggest web search needed
- Confidential terms like "secret", "private", "internal", "confidential" indicate privacy concerns
- Technical terms with "latest", "new version", "update" often need web search
- Questions that could benefit from both sources should be marked as "mixed"

Respond ONLY with valid JSON, no other text:"""

        try:
            response = self.llm.invoke(analysis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response_text)
                
                # Validate and clean the response
                analysis = self._validate_llm_analysis(analysis, query, has_local_docs)
                
                # Build strategy based on LLM analysis
                strategy = self._build_strategy_from_analysis(analysis, has_local_docs)
                
                return {
                    'original_query': query,
                    'privacy_level': analysis.get('privacy_level', 'safe'),
                    'query_type': analysis.get('query_type', 'general_knowledge'),
                    'strategy': strategy,
                    'filtered_query': analysis.get('filtered_query', self._basic_filter_query(query)),
                    'reasoning': analysis.get('reasoning', 'LLM-based analysis'),
                    'llm_analysis': analysis
                }
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                # Try to extract JSON from response if it's embedded in text
                return self._extract_json_from_response(response_text, query, has_local_docs)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analyze_query(query, has_local_docs)

    def _validate_llm_analysis(self, analysis: Dict[str, Any], query: str, has_local_docs: bool) -> Dict[str, Any]:
        """Validate and clean LLM analysis response."""
        
        # Ensure all required fields exist with defaults
        validated = {
            'privacy_level': analysis.get('privacy_level', 'safe'),
            'query_type': analysis.get('query_type', 'general_knowledge'),
            'requires_web_search': analysis.get('requires_web_search', False),
            'requires_local_search': analysis.get('requires_local_search', has_local_docs),
            'priority': analysis.get('priority', 'local' if has_local_docs else 'web'),
            'reasoning': analysis.get('reasoning', 'LLM analysis'),
            'filtered_query': analysis.get('filtered_query', self._basic_filter_query(query))
        }
        
        # Validate enum values
        valid_privacy_levels = ['safe', 'sensitive', 'confidential']
        if validated['privacy_level'] not in valid_privacy_levels:
            validated['privacy_level'] = 'safe'
            
        valid_query_types = ['local_document', 'current_events', 'general_knowledge', 'technical_updates', 'mixed']
        if validated['query_type'] not in valid_query_types:
            validated['query_type'] = 'general_knowledge'
            
        valid_priorities = ['local', 'web', 'hybrid']
        if validated['priority'] not in valid_priorities:
            validated['priority'] = 'local' if has_local_docs else 'web'
        
        return validated

    def _build_strategy_from_analysis(self, analysis: Dict[str, Any], has_local_docs: bool) -> Dict[str, Any]:
        """Build search strategy from LLM analysis."""
        
        strategy = {
            'use_local_search': analysis.get('requires_local_search', has_local_docs),
            'use_web_search': analysis.get('requires_web_search', False),
            'require_user_consent': False,
            'priority': analysis.get('priority', 'local')
        }
        
        # Override web search for confidential queries
        if analysis.get('privacy_level') == 'confidential':
            strategy['use_web_search'] = False
            strategy['priority'] = 'local'
        
        # Require consent for web search unless it's confidential
        if strategy['use_web_search'] and analysis.get('privacy_level') != 'confidential':
            strategy['require_user_consent'] = True
        
        return strategy

    def _extract_json_from_response(self, response_text: str, query: str, has_local_docs: bool) -> Dict[str, Any]:
        """Try to extract JSON from LLM response that might have extra text."""
        
        # Look for JSON-like content in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                analysis = json.loads(match)
                if 'privacy_level' in analysis or 'query_type' in analysis:
                    # Looks like our analysis JSON
                    analysis = self._validate_llm_analysis(analysis, query, has_local_docs)
                    strategy = self._build_strategy_from_analysis(analysis, has_local_docs)
                    
                    return {
                        'original_query': query,
                        'privacy_level': analysis.get('privacy_level', 'safe'),
                        'query_type': analysis.get('query_type', 'general_knowledge'),
                        'strategy': strategy,
                        'filtered_query': analysis.get('filtered_query', self._basic_filter_query(query)),
                        'reasoning': analysis.get('reasoning', 'Extracted from LLM response'),
                        'llm_analysis': analysis
                    }
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, fall back to rule-based
        logger.warning("Could not extract valid JSON from LLM response, using fallback")
        return self._fallback_analyze_query(query, has_local_docs)

    def _basic_filter_query(self, query: str) -> str:
        """Basic query filtering for privacy."""
        filtered_query = query
        
        # Simple filtering patterns
        sensitive_patterns = [
            (r'\b(our|the) (company|organization|team|internal|private)\b', ''),
            (r'\b(according to|based on|in) (our|the) (document|file|data|system)\b', ''),
            (r'\b(confidential|secret|private|internal|proprietary)\b', ''),
            (r'\b(from the (uploaded|attached|provided) (document|file))\b', ''),
            (r'\b(in this (document|file|pdf))\b', ''),
        ]
        
        for pattern, replacement in sensitive_patterns:
            filtered_query = re.sub(pattern, replacement, filtered_query, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        filtered_query = re.sub(r'\s+', ' ', filtered_query).strip()
        
        return filtered_query if filtered_query else query

    def _fallback_analyze_query(self, query: str, has_local_docs: bool) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM is not available."""
        
        query_lower = query.lower()
        
        # Simple rule-based privacy detection
        confidential_keywords = ['confidential', 'secret', 'private', 'internal', 'our company', 'our organization']
        is_confidential = any(keyword in query_lower for keyword in confidential_keywords)
        
        sensitive_keywords = ['our document', 'uploaded file', 'this document', 'according to']
        is_sensitive = any(keyword in query_lower for keyword in sensitive_keywords)
        
        privacy_level = 'confidential' if is_confidential else ('sensitive' if is_sensitive else 'safe')
        
        # Simple query type detection
        current_keywords = ['latest', 'recent', 'current', 'today', 'breaking', 'news']
        tech_keywords = ['version', 'release', 'update', 'changelog']
        doc_keywords = ['document', 'file', 'pdf', 'uploaded']
        
        has_current = any(keyword in query_lower for keyword in current_keywords)
        has_tech = any(keyword in query_lower for keyword in tech_keywords)
        has_doc = any(keyword in query_lower for keyword in doc_keywords)
        
        if has_doc and has_local_docs:
            query_type = 'local_document'
        elif has_current:
            query_type = 'current_events' if not has_doc else 'mixed'
        elif has_tech and has_current:
            query_type = 'technical_updates'
        else:
            query_type = 'general_knowledge'
        
        # Build strategy
        use_web = has_current or has_tech
        use_local = has_local_docs and (has_doc or not use_web)
        
        if is_confidential:
            use_web = False
            
        strategy = {
            'use_local_search': use_local,
            'use_web_search': use_web and not is_confidential,
            'require_user_consent': use_web and not is_confidential,
            'priority': 'local' if use_local and not use_web else ('web' if use_web and not use_local else 'hybrid')
        }
        
        return {
            'original_query': query,
            'privacy_level': privacy_level,
            'query_type': query_type,
            'strategy': strategy,
            'filtered_query': self._basic_filter_query(query),
            'reasoning': f"Rule-based analysis: {query_type} query with {privacy_level} privacy level"
        }

    def should_use_local_search(self, analysis: Dict[str, Any]) -> bool:
        """Determine if local search should be used."""
        return analysis['strategy']['use_local_search']

    def should_use_web_search(self, analysis: Dict[str, Any]) -> bool:
        """Determine if web search should be used."""
        return analysis['strategy']['use_web_search']

    def requires_user_consent(self, analysis: Dict[str, Any]) -> bool:
        """Determine if user consent is required for web search."""
        return analysis['strategy']['require_user_consent']

    def get_filtered_query(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Get the filtered query for web search."""
        return analysis.get('filtered_query')

    def should_use_local_search(self, analysis: Dict[str, Any]) -> bool:
        """Determine if local search should be used."""
        return analysis['strategy']['use_local_search']

    def should_use_web_search(self, analysis: Dict[str, Any]) -> bool:
        """Determine if web search should be used."""
        return analysis['strategy']['use_web_search']

    def requires_user_consent(self, analysis: Dict[str, Any]) -> bool:
        """Determine if user consent is required for web search."""
        return analysis['strategy']['require_user_consent']

    def get_filtered_query(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Get the filtered query for web search."""
        return analysis.get('filtered_query')
