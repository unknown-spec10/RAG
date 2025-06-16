"""RAG Agent implementation using LangGraph for orchestrating the RAG workflow."""
from typing import List, Dict, Any, Optional, TypedDict
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.rag.chroma_retriever import ChromaRetriever
from src.rag.rag.context_protocol import ContextProtocolManager, ContextSet

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

class AgentState(TypedDict):
    """State for the RAG agent workflow."""
    query: str
    context: Optional[ContextSet]
    documents: List[Dict[str, Any]]
    response: Optional[str]
    followup_questions: Optional[List[str]]
    messages: list
    error: Optional[str]
    sources: Optional[List[Dict[str, Any]]]

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
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG agent.
        
        Args:
            retriever: ChromaDB retriever instance.
            model_name: Name of the LLM to use.
            temperature: Temperature for LLM generation.
            context_window: Context window size for the LLM.
            max_tokens: Maximum tokens to generate.
            context_protocol: Context protocol manager.
            api_key: Groq API key. If not provided, will look for GROQ_API_KEY environment variable.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.context_protocol = context_protocol or ContextProtocolManager()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
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
        """Generate a response based on the retrieved documents."""
        query = state.get("query", "")
        documents = state.get("documents", [])
        
        # Format the prompt for the LLM
        prompt = self._format_rag_prompt(query, documents)
        
        # Generate response using the LLM
        response = self.llm.invoke(prompt)
        
        # Parse the response to separate the answer from the source chunks
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Split the response into answer and sources
        parts = response_text.split("Source:")
        answer = parts[0].strip()
        sources = []
        
        # Process the source chunks
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
                
            # Try to extract source information
            source_info = {}
            
            # Look for source name in quotes
            if '"' in part:
                quote_parts = part.split('"', 2)
                if len(quote_parts) >= 3:
                    source_info['source'] = quote_parts[1].strip()
                    part = quote_parts[2].strip()
            
            # Look for page number
            if "(Page" in part:
                page_parts = part.split("(Page", 1)
                if len(page_parts) >= 2:
                    if not source_info.get('source'):
                        source_info['source'] = page_parts[0].strip()
                    page_info = page_parts[1].split(")", 1)
                    source_info['page'] = page_info[0].strip()
                    part = page_info[1].strip() if len(page_info) > 1 else ""
            
            # If we found source info, add it to sources
            if source_info:
                source_info['text'] = part.strip()
                sources.append(source_info)
            else:
                # If no clear source format, add as is
                    sources.append({
                    'source': 'Unknown',
                    'page': 'Unknown',
                    'text': part.strip()
                    })
        
        # Update state with the response and sources
        state["response"] = answer
        state["sources"] = sources
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
    
    def _format_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format the prompt for the RAG system."""
        # Sort documents by similarity score
        sorted_docs = sorted(documents, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Format context with clear source attribution
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

    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the RAG system.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary containing the response, sources, and follow-up questions
        """
        # Initialize the state
        state = {
            "query": query,
            "context": None,
            "documents": [],
            "response": None,
            "followup_questions": None,
            "messages": [],
            "error": None,
            "sources": None
        }
        
        try:
            # Run the workflow
            final_state = self.graph.invoke(state)
            
            # Return the results
            return {
                "response": final_state["response"],
                "sources": final_state["sources"],
                "followup_questions": final_state["followup_questions"]
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "followup_questions": []
            }
