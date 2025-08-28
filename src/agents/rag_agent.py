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
        tools: Optional[List[Any]] = None  # Add tools argument
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
