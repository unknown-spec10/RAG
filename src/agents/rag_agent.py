"""RAG Agent implementation using LangGraph for orchestrating the RAG workflow."""
from typing import List, Dict, Any, Optional, Union, Callable, TypedDict, Annotated
import json
import os
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
# Try to import Ollama, but don't fail if it's not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Please install Ollama or use a different provider.")

try:
    from groq import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq client not available. Install 'groq' package to use Groq API.")

from src.rag.retriever import RAGRetriever
from src.rag.context_protocol import ContextProtocolManager, ContextSet


class AgentState(TypedDict):
    """State for the RAG agent workflow."""
    
    query: str
    context: Optional[ContextSet] 
    documents: List[Dict[str, Any]]
    response: Optional[str]
    followup_questions: Optional[List[str]]
    messages: List[BaseMessage]
    error: Optional[str]


class RAGAgent:
    """Agent for orchestrating the RAG workflow using LangGraph."""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        context_window: int = 8192,
        max_tokens: int = 1024,
        context_protocol: Optional[ContextProtocolManager] = None,
    ):
        """
        Initialize the RAG agent.
        
        Args:
            retriever: RAG retriever instance.
            model_name: Name of the LLM to use (OpenAI model or Ollama model).
            temperature: Temperature for LLM generation.
            context_window: Context window size for the LLM.
            max_tokens: Maximum tokens to generate.
            context_protocol: Context protocol manager.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.context_protocol = context_protocol or ContextProtocolManager()
        
        # Configure LLM based on model name
        self.llm = self._initialize_llm(model_name, temperature, max_tokens)
        
        # Initialize the workflow
        self.graph = self._build_graph()
        
    def _initialize_llm(self, model_name: str, temperature: float, max_tokens: int) -> Any:
        """
        Initialize the appropriate LLM based on the model name and provider.
        
        Args:
            model_name: Name of the model to use.
            temperature: Temperature setting for generation.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Initialized LLM instance.
        """
        # Get the LLM provider from config
        config = Config()
        provider = config.get("llm", "provider")
        
        # Initialize based on provider
        if provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError(
                    "Ollama is not available. Please install Ollama or use a different provider."
                )
            return ollama.Client()
        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError(
                    "Groq client is not available. Please install 'groq' package or use a different provider."
                )
            
            # Get Groq API key from config
            groq_api_key = config.get("llm", "groq_api_key")
            if not groq_api_key:
                raise ValueError("Groq API key must be provided in configuration")
            
            return GroqClient(
                api_key=groq_api_key,
                model=config.get("llm", "groq_model")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
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
        
        # Compile the graph
        return workflow.compile()
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Node for retrieving relevant documents.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated agent state.
        """
        query = state["query"]
        
        try:
            # Retrieve relevant documents
            documents = self.retriever.retrieve(query=query, rerank=True)
            
            # Create context set using the context protocol
            context_set = self.context_protocol.from_retrieved_documents(
                documents=documents,
                query=query
            )
            
            # Update state
            return {
                **state,
                "documents": documents,
                "context": context_set
            }
        except Exception as e:
            # Handle errors
            return {
                **state,
                "error": f"Error retrieving documents: {str(e)}"
            }
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Node for generating a response based on retrieved documents.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated agent state.
        """
        query = state["query"]
        context_set = state.get("context")
        
        if not context_set or state.get("error"):
            # Handle case with no context or errors
            response = "I couldn't find relevant information to answer your question."
            if state.get("error"):
                response += f" Error: {state['error']}"
            
            # Set default messages
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=query))
            messages.append(AIMessage(content=response))
            
        else:
            try:
                # Format prompt
                prompt = self._format_rag_prompt(query, context_set)
                
                # Different handling based on LLM type
                if isinstance(self.llm, ChatOpenAI):
                    # Using OpenAI
                    response = self.llm.invoke([
                        {"role": "system", "content": "You are a helpful RAG assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ]).content
                elif isinstance(self.llm, GroqClient):
                    # Using Groq
                    response = self.llm.generate(
                        messages=[
                            {"role": "system", "content": "You are a helpful RAG assistant that answers questions based on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Extract response from Groq's response structure
                    response = response["choices"][0]["message"]["content"]
                else:
                    # Using Ollama
                    response = self.llm.generate(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful RAG assistant that answers questions based on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Extract response from Ollama's response structure
                    response = response["response"]
                
                # Add to messages
                messages = state.get("messages", [])
                messages.append(HumanMessage(content=query))
                messages.append(AIMessage(content=response))
                
            except Exception as e:
                response = f"An error occurred while generating the response: {str(e)}"
                messages = state.get("messages", [])
                messages.append(HumanMessage(content=query))
                messages.append(AIMessage(content=response))
        
        # Update state
        return {
            **state,
            "response": response,
            "messages": messages
        }
    
    def _extract_followup_questions(self, state: AgentState) -> AgentState:
        """
        Node for extracting potential follow-up questions.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated agent state.
        """
        response = state.get("response", "")
        query = state["query"]
        followup_questions = []
        
        try:
            # Prompt for follow-up questions
            followup_prompt = f"""
            Based on the user's question: "{query}" 
            And the response that was provided: "{response}"
            
            Generate 3 potential follow-up questions that the user might ask next.
            Format your response as a JSON list of strings.
            """
            
            # Different handling based on LLM type
            if isinstance(self.llm, ChatOpenAI):
                # Using OpenAI
                followup_text = self.llm.invoke([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": followup_prompt}
                ]).content
            else:
                # Using Ollama
                llm_response = self.llm.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": followup_prompt}
                    ],
                    options={
                        "temperature": 0.7,  # Higher temperature for creativity
                        "num_predict": 500,
                    }
                )
                followup_text = llm_response["message"]["content"]
            
            # Try to parse as JSON
            try:
                # Extract JSON part if it exists
                if "```json" in followup_text:
                    json_str = followup_text.split("```json")[1].split("```")[0].strip()
                elif "```" in followup_text:
                    json_str = followup_text.split("```")[1].strip()
                else:
                    json_str = followup_text
                    
                # Try to parse JSON
                followup_questions = json.loads(json_str)
            except Exception:
                # Fallback: try to extract questions heuristically
                import re
                followup_questions = []
                for line in followup_text.split("\n"):
                    line = line.strip()
                    # Look for lines that appear to be questions
                    if line and ("?" in line) and len(line) > 10:
                        # Clean up formatting like numbers or bullet points
                        clean_line = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                        if clean_line:
                            followup_questions.append(clean_line)
        except Exception as e:
            # If there's an error, log it and return empty list
            logging.warning(f"Error generating follow-up questions: {str(e)}")
            followup_questions = []
        
        # Ensure we have a list of strings and limit to 3
        if not isinstance(followup_questions, list):
            followup_questions = []
        followup_questions = [q for q in followup_questions if isinstance(q, str)][:3]
        
        # Update state
        return {
            **state,
            "followup_questions": followup_questions
        }
    
    def _format_rag_prompt(self, query: str, context_set: ContextSet) -> str:
        """
        Format the RAG prompt with query and context.
        
        Args:
            query: User query.
            context_set: Context set with retrieved documents.
            
        Returns:
            Formatted prompt.
        """
        prompt = f"""
        Answer the question below based only on the provided context. If the context doesn't contain the information needed, say "I don't have enough information to answer this question" and don't make up an answer.
        
        Context:
        {context_set.to_llm_context()}
        
        Question: {query}
        
        Answer:
        """
        
        return prompt
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG agent.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary with response and other outputs.
        """
        # Initialize state
        initial_state = {
            "query": query,
            "context": None,
            "documents": [],
            "response": None,
            "followup_questions": None,
            "messages": [],
            "error": None
        }
        
        # Run the workflow
        result = self.graph.invoke(initial_state)
        
        # Format the output
        output = {
            "query": query,
            "response": result["response"],
            "followup_questions": result.get("followup_questions", []),
            "sources": [
                {
                    "content": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                }
                for doc in result.get("documents", [])
            ],
            "error": result.get("error")
        }
        
        return output
