"""RAG Agent implementation using LangGraph for orchestrating the RAG workflow."""
from typing import List, Dict, Any, Optional, TypedDict
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.rag.retriever import RAGRetriever
from src.rag.context_protocol import ContextProtocolManager, ContextSet

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
        retriever: RAGRetriever,
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
            retriever: RAG retriever instance.
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
        documents = state["documents"]
        
        # Retrieve relevant documents using the retriever
        retrieved_docs = self.retriever.retrieve(query, documents)
        
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
            if '"' in part:
                # Extract source name and text
                source_parts = part.split('"', 2)
                if len(source_parts) >= 3:
                    source_name = source_parts[1].strip()
                    # Extract page number if present
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
        
        followup_response = self.llm.invoke(prompt)
        
        # Parse follow-up questions
        questions = []
        if hasattr(followup_response, 'content'):
            questions = [q.strip() for q in followup_response.content.split('\n') if q.strip()]
        else:
            questions = ["The answer is not available in your policy or documents."]
        
        # Update state with follow-up questions
        state["followup_questions"] = questions
        return state
    
    def _format_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format the prompt for the LLM with strict context-based response rules."""
        system_message = """You are a context-aware assistant. Follow these rules strictly:
1. ONLY answer using the provided context.
2. If the answer is not present in the context, respond with: "The answer is not available in your policy or documents."
3. DO NOT use any external knowledge.
4. DO NOT make assumptions or fabricate information.
5. DO NOT try to be helpful by guessing or providing general information.
6. If the context is insufficient, simply state that the information is not available.
7. For each piece of information you use, reference which source it came from using Source: "filename" (Page X).
8. After your response, list ONLY the specific chunks you used to answer the question, in this format:
   Source: "filename" (Page X)
   [Relevant chunk text]"""

        prompt = f"{system_message}\n\nContext:\n"
        for i, doc in enumerate(documents, 1):
            source_name = doc.get('metadata', {}).get('source', 'Unknown')
            page_num = doc.get('metadata', {}).get('page', 'Unknown')
            prompt += f"[Source {i}: {source_name} (Page {page_num})]\n{doc.get('text', '')}\n\n"
        prompt += f"\nQuestion: {query}\n\nAnswer (based ONLY on the above context, include source references):"
        return prompt
    
    def query(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a query using the RAG workflow.
        
        Args:
            query: Query string.
            documents: List of documents to search in.
            
        Returns:
            Dictionary containing the response, follow-up questions, and sources.
        """
        # Initialize the state
        state = {
            "query": query,
            "documents": documents,
            "response": None,
            "followup_questions": None,
            "messages": [],
            "error": None,
            "sources": None
        }
        
        # Run the workflow
        result = self.graph.invoke(state)
        
        return {
            "response": result["response"],
            "followup_questions": result["followup_questions"],
            "sources": result["sources"]  # Include sources in the response
        }
