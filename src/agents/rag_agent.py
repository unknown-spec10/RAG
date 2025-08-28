"""RAG Agent implementation using LangGraph for orchestrating the RAG workflow."""
from typing import List, Dict, Any, Optional, TypedDict
import logging
import os
import uuid
import json
from datetime import datetime
from typing import Union

# Configure standard logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.rag.chroma_retriever import ChromaRetriever
from src.rag.rag.context_protocol import ContextProtocolManager, ContextSet
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.rerankers import JinaReranker
from src.rag.rag.retriever import RAGRetriever
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.web_search_agent import WebSearchAgent


class MockLLM(BaseChatModel):
    """Mock LLM for testing and fallback purposes."""
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response."""
        last_message = messages[-1].content if messages else ""
        
        if "follow-up questions" in last_message.lower():
            response_text = "1. Can you provide more details about that?\n2. What are the key aspects to consider?\n3. How does this compare to other approaches?"
        else:
            response_text = "Based on the provided context, I can provide a comprehensive response. The information suggests that this is a relevant topic that requires careful consideration of various factors."
        
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

class AgentState(TypedDict, total=False):
    """State for the RAG agent workflow, supporting multi-turn conversations and session tracking."""
    query: str
    context: Optional[ContextSet]
    documents: List[Dict[str, Any]]
    response: Optional[str]
    followup_questions: Optional[List[str]]
    messages: list
    error: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
    chat_id: str  # Unique session ID
    history: list  # List of dicts with previous queries, answers, docs
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp


class RAGAgent:
    """Agent for orchestrating the RAG workflow using LangGraph."""
    
    def __init__(
        self,
        retriever: ChromaRetriever,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.1,
        context_window: int = 8192,
        max_tokens: int = 1024,
        context_protocol: Optional[ContextProtocolManager] = None,
        api_key: Optional[str] = None,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.5,
        use_reranker: bool = True,
        reranker_model: str = "jina-reranker-v1-base-en",
        tools: Optional[List[Any]] = None,  # Add tools argument
        serp_api_key: Optional[str] = None  # Add SERP API key
    ):
        """
        Initialize the RAG agent with optional hybrid retrieval and re-ranking.
        Args:
            retriever: ChromaDB retriever instance.
            model_name: Name of the LLM to use.
            temperature: Temperature for LLM generation.
            context_window: Context window size for the LLM.
            max_tokens: Maximum tokens to generate.
            context_protocol: Context protocol manager.
            api_key: Groq API key. If not provided, will look for GROQ_API_KEY environment variable.
            use_hybrid: Enable hybrid retrieval (vector + keyword)
            hybrid_alpha: Weight for blending vector and keyword scores
            use_reranker: Enable Jina re-ranking
            reranker_model: Jina model name for re-ranking
            tools: List of tools to integrate with the agent (prototype)
            serp_api_key: SERP API key for web search functionality
        """
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.context_protocol = context_protocol or ContextProtocolManager()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.tools = tools  # Store tools for future integration
        
        # Initialize orchestrator and web search agents
        self.orchestrator = OrchestratorAgent(api_key=self.api_key, model_name=model_name)
        self.web_search_agent = WebSearchAgent(serp_api_key)

        # --- Hybrid and Reranker Setup ---
        if use_hybrid:
            # Set up keyword retriever (TF-IDF)
            self.keyword_retriever = RAGRetriever()
            # You must add documents to self.keyword_retriever elsewhere!
            self.reranker = JinaReranker(model_name=reranker_model) if use_reranker else None
            self.retriever = HybridRetriever(
                chroma_retriever=retriever,
                tfidf_retriever=self.keyword_retriever,
                reranker=self.reranker,
                alpha=hybrid_alpha
            )
        else:
            self.retriever = retriever
            self.reranker = None
        # ---

        # Configure LLM
        self.llm = self._initialize_llm()
        # Initialize the workflow
        self.graph = self._build_graph()

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the LLM (either Groq or Mock)."""
        if self.api_key:
            try:
                return ChatGroq(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            except Exception as e:
                logging.warning(f"Failed to initialize Groq LLM: {e}. Falling back to mock LLM.")
                return MockLLM()
        else:
            logging.warning("No API key provided. Using mock LLM.")
            return MockLLM()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("extract_followup", self._extract_followup_questions)
        
        # Define edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "extract_followup")
        workflow.add_edge("extract_followup", END)
        
        # Set the entry point
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents for the query."""
        query = state["query"]
        
        # Retrieve relevant documents using the retriever
        retrieved_docs = self.retriever.retrieve(query)
        
        # Update state with retrieved documents
        state["documents"] = retrieved_docs
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response based on the context set (dynamic context sizing), with confidence scoring."""
        query = state.get("query", "")
        documents = state.get("documents", [])
        context_set = state.get("context")
        # Format the prompt for the LLM
        prompt = self._format_rag_prompt(query, documents, context_set=context_set)
        # Add a confidence scoring instruction
        prompt += "\n\nAfter your answer, provide a confidence score (0-1) based ONLY on how much you relied on the provided context versus your own knowledge. Format: CONFIDENCE: <score>"
        # Generate response using the LLM
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # Parse confidence score
        confidence = None
        if "CONFIDENCE:" in response_text:
            main, conf_part = response_text.rsplit("CONFIDENCE:", 1)
            try:
                confidence = float(conf_part.strip().split()[0])
            except Exception:
                confidence = None
            response_text = main.strip()
        # Split the response into answer and sources
        parts = response_text.split("Source:")
        answer = parts[0].strip()
        sources = []
        for part in parts[1:]:
            if '"' in part:
                source_parts = part.split('"', 2)
                if len(source_parts) >= 3:
                    source_name = source_parts[1].strip()
                    page_num = "Unknown"
                    if "Page" in source_name:
                        source_name, page_info = source_name.split("(Page", 1)
                        source_name = source_name.strip()
                        page_num = page_info.strip().rstrip(")")
                    chunk_text = source_parts[2].strip()
                    sources.append({
                        "source": source_name,
                        "text": chunk_text,
                        "page": page_num
                    })
        state["response"] = answer
        state["sources"] = sources
        state["confidence"] = confidence
        return state

    
    def _extract_followup_questions(self, state: AgentState) -> AgentState:
        """Extract follow-up questions based only on the available context."""
        response = state["response"]
        documents = state["documents"]
        
        # Generate follow-up questions using the LLM
        prompt = f"""Based ONLY on the provided context, generate 2-3 relevant follow-up questions.
Rules:
1. Questions must be answerable using ONLY the provided context
2. Do not generate questions about topics not covered in the context
3. Focus on clarifying or expanding on information that is actually present

Context:
{chr(10).join([doc['text'] for doc in documents])}

Current Response:
{response}

Generate 2-3 follow-up questions that can be answered using ONLY the above context:"""
        
        # Generate follow-up questions
        followup_response = self.llm.invoke(prompt)
        followup_text = followup_response.content if hasattr(followup_response, 'content') else str(followup_response)
        
        # Parse follow-up questions
        questions = []
        for line in followup_text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                question = line.split('. ', 1)[1].strip()
                questions.append(question)
        
        # Update state with follow-up questions
        state["followup_questions"] = questions
        return state
    
    def _format_rag_prompt(self, query: str, documents: List[Dict[str, Any]], context_set: Optional[ContextSet] = None) -> str:
        """
        Format the prompt for the RAG system, optionally using a ContextSet for advanced context management.
        """
        if context_set is not None:
            context = context_set.to_formatted_string(include_metadata=True)
        else:
            # Fallback to legacy formatting
            sorted_docs = sorted(documents, key=lambda x: x.get('similarity_score', 0), reverse=True)
            context_parts = []
            for doc in sorted_docs:
                source = doc.get('metadata', {}).get('source', 'Unknown')
                page = doc.get('metadata', {}).get('page', 'Unknown')
                text = doc.get('text', '')
                context_parts.append(f"Source: {source} (Page {page})\n{text}\n")
            context = "\n".join(context_parts)
        return f"""You are a helpful AI assistant. Use the following context to answer the question.
If you cannot answer the question using only the provided context, say so.
Always cite your sources using the format: Source: "source_name (Page X)" followed by the relevant text.

Rules for answering:
1. Provide a complete and comprehensive answer using ALL relevant information from the context
2. If the answer spans multiple sources, combine them into a coherent response
3. Include ALL relevant steps or details, not just the first ones you find
4. If the information seems incomplete, mention that there might be more details available
5. Always cite your sources for each piece of information

Context:
{context}

Question: {query}

Provide a comprehensive answer based ONLY on the provided context. Include relevant source citations."""

    def query_with_orchestration(self, query: str, chat_id: str = None, user_consent_callback=None) -> Dict[str, Any]:
        """
        Process a query using orchestrated search strategy with privacy protection.
        
        Args:
            query: The user query
            chat_id: Optional session ID
            user_consent_callback: Function to get user consent for web search
            
        Returns:
            Dictionary with response, sources, analysis, and follow-up questions
        """
        # Step 1: Analyze query with orchestrator
        has_local_docs = hasattr(self.retriever, 'doc_count') and self.retriever.doc_count > 0
        analysis = self.orchestrator.analyze_query(query, has_local_docs)
        
        logger.info(f"Query analysis: {analysis['reasoning']}")
        
        # Step 2: Check if user consent is required for web search
        if analysis['strategy']['require_user_consent'] and analysis['strategy']['use_web_search']:
            if user_consent_callback:
                consent_granted = user_consent_callback(analysis)
                if not consent_granted:
                    # Use only local search
                    analysis['strategy']['use_web_search'] = False
                    logger.info("User declined web search, using local search only")
            else:
                # No consent callback provided, skip web search
                analysis['strategy']['use_web_search'] = False
                logger.info("No consent callback provided, skipping web search")
        
        # Step 3: Gather information based on strategy
        local_results = []
        web_results = []
        
        # Local search
        if analysis['strategy']['use_local_search']:
            try:
                local_results = self.retriever.retrieve(query, top_k=8)
                logger.info(f"Retrieved {len(local_results)} local documents")
            except Exception as e:
                logger.error(f"Local search failed: {str(e)}")
        
        # Web search
        if analysis['strategy']['use_web_search'] and self.web_search_agent.is_available():
            try:
                filtered_query = analysis['filtered_query']
                web_results = self.web_search_agent.search_general(filtered_query)
                logger.info(f"Retrieved {len(web_results)} web results for query: {filtered_query}")
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
        
        # Step 4: Combine and process results
        combined_context = self._combine_local_and_web_results(local_results, web_results, analysis)
        
        # Step 5: Generate response using combined context
        if combined_context:
            response_result = self._generate_response_with_context(query, combined_context, chat_id)
        else:
            response_result = {
                "response": "I couldn't find sufficient information to answer your question. Please try rephrasing your query or providing more context.",
                "sources": [],
                "followup_questions": []
            }
        
        # Step 6: Add analysis information to response
        response_result.update({
            "query_analysis": {
                "privacy_level": analysis['privacy_level'],
                "query_type": analysis['query_type'],
                "strategy_used": analysis['strategy'],
                "reasoning": analysis['reasoning'],
                "filtered_query": analysis.get('filtered_query'),
                "local_results_count": len(local_results),
                "web_results_count": len(web_results)
            }
        })
        
        return response_result

    def _combine_local_and_web_results(self, local_results: List[Dict], web_results: List[Dict], analysis: Dict) -> str:
        """Combine local and web search results into a coherent context."""
        combined_parts = []
        
        # Add local results
        if local_results:
            local_context = "Local Document Information:\n\n"
            for i, result in enumerate(local_results[:5]):  # Limit to top 5
                local_context += f"{i+1}. {result.get('text', '')[:500]}...\n"
                local_context += f"   Source: {result.get('metadata', {}).get('source', 'Unknown')}\n\n"
            combined_parts.append(local_context)
        
        # Add web results
        if web_results:
            web_context = "Current Web Information:\n\n"
            web_context += self.web_search_agent.format_results_for_context(web_results, max_length=1500)
            combined_parts.append(web_context)
        
        # Combine with priority based on strategy
        if analysis['strategy']['priority'] == 'local':
            return "\n".join(combined_parts)
        elif analysis['strategy']['priority'] == 'web':
            return "\n".join(reversed(combined_parts))
        else:  # hybrid
            return "\n".join(combined_parts)

    def _generate_response_with_context(self, query: str, context: str, chat_id: str = None) -> Dict[str, Any]:
        """Generate response using the provided context."""
        # Use existing query method but with the combined context
        # This is a simplified approach - in production, you might want more sophisticated context handling
        
        # Create a mock state with the combined context
        session_id = chat_id or str(uuid.uuid4())
        
        try:
            # Format the context for the LLM
            formatted_context = f"""Based on the following information, please answer the user's question:

{context}

Question: {query}

Please provide a comprehensive answer using the information provided above. If the information comes from web sources, mention that it's from current web search. If it's from local documents, indicate that as well."""

            # Generate response using LLM
            response = self.llm.invoke(formatted_context)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract sources from context (simplified)
            sources = self._extract_sources_from_context(context)
            
            # Generate follow-up questions
            followup_questions = self._generate_followup_questions(query, response_text, context)
            
            return {
                "response": response_text,
                "sources": sources,
                "followup_questions": followup_questions
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "followup_questions": []
            }

    def _extract_sources_from_context(self, context: str) -> List[Dict[str, Any]]:
        """Extract source information from the combined context."""
        sources = []
        
        # Simple extraction based on context structure
        lines = context.split('\n')
        current_source = {}
        
        for line in lines:
            if line.strip().startswith('Source:'):
                if current_source:
                    sources.append(current_source)
                current_source = {
                    "source": line.replace('Source:', '').strip(),
                    "text": "",
                    "type": "local"
                }
            elif line.strip().startswith('http'):
                current_source = {
                    "source": line.strip(),
                    "text": "",
                    "type": "web"
                }
            elif current_source and line.strip() and not line.startswith('   '):
                current_source["text"] = line.strip()[:200] + "..."
        
        if current_source:
            sources.append(current_source)
        
        return sources[:5]  # Limit to 5 sources

    def _generate_followup_questions(self, query: str, response: str, context: str) -> List[str]:
        """
        Generate follow-up questions based on the query, response, and context.
        
        Args:
            query: Original user query
            response: Generated response
            context: Context used for generation
            
        Returns:
            List of follow-up questions
        """
        try:
            # Create a prompt for generating follow-up questions
            followup_prompt = f"""Based on the following conversation, generate 3 relevant follow-up questions that the user might want to ask.

Original Question: {query}

Response: {response[:500]}...

Context: {context[:800]}...

Generate 3 specific, relevant follow-up questions that would help the user explore this topic further. Focus on:
1. Clarifying questions about the response
2. Related topics that might be of interest  
3. Deeper dive questions into specific aspects

Format as a simple list, one question per line, without numbering."""

            # Generate follow-up questions using the LLM
            followup_response = self.llm.invoke(followup_prompt)
            followup_text = followup_response.content if hasattr(followup_response, 'content') else str(followup_response)
            
            # Parse the response into individual questions
            questions = []
            for line in followup_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 10:
                    # Remove any numbering or bullet points
                    clean_line = line.lstrip('0123456789.-• ')
                    if clean_line.endswith('?'):
                        questions.append(clean_line)
            
            # Limit to 3 questions and ensure they're not too similar to the original
            unique_questions = []
            for q in questions[:5]:  # Check top 5 candidates
                if len(unique_questions) < 3 and not self._is_similar_question(query, q):
                    unique_questions.append(q)
            
            return unique_questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            # Return some generic follow-up questions as fallback
            return [
                "Can you provide more details about this topic?",
                "What are the key takeaways from this information?",
                "Are there any related topics I should know about?"
            ]

    def _is_similar_question(self, original: str, candidate: str) -> bool:
        """Check if a candidate question is too similar to the original."""
        original_words = set(original.lower().split())
        candidate_words = set(candidate.lower().split())
        
        # Calculate word overlap
        overlap = len(original_words & candidate_words)
        total_unique_words = len(original_words | candidate_words)
        
        # If more than 60% overlap, consider it too similar
        similarity = overlap / total_unique_words if total_unique_words > 0 else 0
        return similarity > 0.6

    def query(self, query: str, chat_id: str = None, persist_state: bool = False) -> Dict[str, Any]:
        """
        Process a query using the RAG system, with robust error handling, logging, and session support.
        Args:
            query: The query to process
            chat_id: Optional session ID for multi-turn conversations
            persist_state: If True, persist AgentState after each step (prototype)
        Returns:
            Dictionary containing the response, sources, and follow-up questions
        """
        session_id = chat_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        state: AgentState = {
            "query": query,
            "context": None,
            "documents": [],
            "response": None,
            "followup_questions": None,
            "messages": [],
            "error": None,
            "sources": None,
            "chat_id": session_id,
            "history": [],
            "created_at": now,
            "updated_at": now
        }
        
        logger.info(f"RAG Agent session start - query: {query}, chat_id: {session_id}")
        try:
            # --- Step 1: Document Retrieval ---
            try:
                logger.info(f"RAG Agent retrieval start - chat_id: {session_id}")
                # HybridRetriever will combine vector and keyword, then rerank via Jina if enabled
                retrieved_docs = self.retriever.retrieve(query, top_k=8)
                
                # Ensure retrieved_docs is not None and is a list
                if retrieved_docs is None:
                    retrieved_docs = []
                elif not isinstance(retrieved_docs, list):
                    retrieved_docs = []
                
                state["documents"] = retrieved_docs
                logger.info(f"RAG Agent retrieval success - chat_id: {session_id}, docs: {len(retrieved_docs)}")
                
                # If no documents retrieved, return early with appropriate message
                if not retrieved_docs:
                    state["response"] = "I couldn't find any relevant information to answer your question. Please try rephrasing your query or upload a different document."
                    state["sources"] = []
                    state["followup_questions"] = []
                    if persist_state:
                        self._persist_state(state)
                    return self._finalize_state(state)
                
            except Exception as e:
                logger.error(f"RAG Agent retrieval error - chat_id: {session_id}, error: {str(e)}")
                state["error"] = f"Retrieval error: {str(e)}"
                if persist_state:
                    self._persist_state(state)
                return self._finalize_state(state)
            # --- Step 2: Context Management and Workflow ---
            try:
                logger.info(f"RAG Agent context start - chat_id: {session_id}")
                # Step 2a: Build initial context set from retrieved docs
                context_set = self.context_protocol.from_retrieved_documents(state["documents"], query=query)

                # Step 2b: Refine protocol (e.g., prioritize core facts)
                context_set = self.context_protocol.refine_protocol(context_set, protocol="core-first")

                # Step 2c: Dynamic context sizing (fit for LLM window)
                # Use a tokenizer compatible with the LLM (user should provide or fallback to .split)
                try:
                    import tiktoken
                    tokenizer = tiktoken.encoding_for_model(self.model_name)
                except Exception:
                    tokenizer = type("DummyTokenizer", (), {"encode": lambda self, x: x.split()})()
                max_tokens = self.context_window
                expected_response_tokens = self.max_tokens
                context_set = context_set.prioritize_and_truncate(tokenizer, max_tokens, query, expected_response_tokens)

                # Step 2d: Condense context if still too long
                if context_set.token_length(tokenizer) + expected_response_tokens > max_tokens:
                    context_set = context_set.condense(self.llm, max_tokens - expected_response_tokens, tokenizer)

                state["context"] = context_set
                logger.info(f"RAG Agent context success - chat_id: {session_id}, items: {len(context_set.items)}")

                # Step 2e: Run workflow (graph)
                final_state = self.graph.invoke(state)
                logger.info(f"RAG Agent workflow success - chat_id: {session_id}")
            except Exception as e:
                logger.error(f"RAG Agent workflow error - chat_id: {session_id}, error: {str(e)}")
                state["error"] = f"Workflow error: {str(e)}"
                if persist_state:
                    self._persist_state(state)
                return self._finalize_state(state)
            # --- Step 3: Multi-turn History ---
            state["history"].append({
                "query": query,
                "response": final_state.get("response"),
                "documents": final_state.get("documents"),
                "timestamp": now
            })
            state.update(final_state)
            state["updated_at"] = datetime.utcnow().isoformat()
            if persist_state:
                self._persist_state(state)
            logger.info(f"RAG Agent session success - chat_id: {session_id}, response: {final_state.get('response', '')[:100]}...")
            return self._finalize_state(state)
        except Exception as e:
            logger.error(f"RAG Agent session error - chat_id: {session_id}, error: {str(e)}")
            state["error"] = f"Session error: {str(e)}"
            if persist_state:
                self._persist_state(state)
            return self._finalize_state(state)

    def _persist_state(self, state: AgentState, db_path: str = "rag_sessions.json"):
        """Persist AgentState to a JSON file (prototype for production DB/Redis)."""
        try:
            if os.path.exists(db_path):
                with open(db_path, "r") as f:
                    sessions = json.load(f)
            else:
                sessions = {}
            sessions[state["chat_id"]] = state
            with open(db_path, "w") as f:
                json.dump(sessions, f, indent=2)
            logger.info(f"RAG Agent state persisted - chat_id: {state['chat_id']}")
        except Exception as e:
            logger.error(f"RAG Agent state persist error - chat_id: {state.get('chat_id')}, error: {str(e)}")

    def _finalize_state(self, state: AgentState) -> Dict[str, Any]:
        """Return the final state with required fields for output."""
        return {
            "response": state.get("response"),
            "sources": state.get("sources", []),
            "followup_questions": state.get("followup_questions", []),
            "chat_id": state.get("chat_id"),
            "history": state.get("history", []),
            "error": state.get("error")
        }

    # --- Async Support (Prototype) ---
    # import asyncio
    # async def query_async(...):
    #     ...
