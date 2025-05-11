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
    from groq import Groq
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
        model_name: str = "llama3-70b-8192",
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
        try:
            # Use native Groq client directly
            from groq import Groq
            import os
            
            # Get Groq API key from environment variable
            groq_api_key = None
            try:
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    groq_api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
                
            # If not in Streamlit secrets, try environment variable
            if not groq_api_key:
                groq_api_key = os.environ.get("GROQ_API_KEY")
            
            # If still not found, try .env file
            if not groq_api_key:
                from dotenv import load_dotenv
                load_dotenv()
                groq_api_key = os.environ.get("GROQ_API_KEY")
            
            # If still not found, use the example key from secrets.toml.example
            if not groq_api_key:
                # This is the key from secrets.toml.example
                groq_api_key = "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"
                
            # Create a simple wrapper class for Groq client to make it compatible with our interface
            class GroqWrapper:
                def __init__(self, client, model_name, temperature, max_tokens):
                    self.client = client
                    # Map common model names to current Groq-supported models
                    # Based on https://console.groq.com/docs/models
                    model_mapping = {
                        "mixtral": "mixtral-8x7b-32768",  # WARNING: Decommissioned - will be replaced
                        "llama3": "llama3-70b-8192",      # Current flagship model
                        "llama3-small": "llama3-8b-8192", # Smaller but faster model
                        "llama2": "llama2-70b-4096",      # Legacy but stable model
                        "gemma": "gemma-7b-it"            # Google's smaller model
                    }
                    
                    # Special handling for decommissioned models
                    if model_name in ["mixtral", "mixtral-8x7b", "mixtral-8x7b-32768"]:
                        print("⚠️ Warning: Mixtral models have been decommissioned on Groq.")
                        print("Using LLama3-70b instead.")
                        self.model_name = "llama3-70b-8192"
                    else:
                        # Use the mapped model name if available, otherwise keep the original
                        self.model_name = model_mapping.get(model_name, model_name)
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                    print(f"Using Groq model: {self.model_name}")
                
                def invoke(self, messages):
                    # Convert to format expected by Groq
                    formatted_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            formatted_messages.append(msg)
                        else:
                            # Handle other message formats if needed
                            formatted_messages.append({"role": "user", "content": str(msg)})
                    
                    # Call Groq API
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return completion.choices[0].message.content
            
            # Initialize the Groq client
            client = Groq(api_key=groq_api_key)
            return GroqWrapper(
                client=client,
                model_name=model_name,  # Use the model name passed to initialize_llm
                temperature=temperature,
                max_tokens=max_tokens
            )
        except (ImportError, ValueError) as e:
            # If Groq is not available or API key is missing, use a simple mock LLM
            from langchain.llms.fake import FakeListLLM
            return FakeListLLM(
                responses=[f"Error initializing Groq LLM: {str(e)}. This is a demo response."]
            )
    
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
                
                # Generic approach that works with any LLM type
                try:
                    # First try the invoke method (new LangChain API)
                    try:
                        system_message = {"role": "system", "content": "You are a helpful RAG assistant that answers questions based on the provided context."}
                        user_message = {"role": "user", "content": prompt}
                        result = self.llm.invoke([system_message, user_message])
                        
                        # Extract response based on result type
                        if hasattr(result, 'content'):
                            response = result.content  # LangChain standard response
                        elif isinstance(result, dict) and 'content' in result:
                            response = result["content"]
                        elif isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
                            # Groq-like response structure
                            response = result["choices"][0]["message"]["content"]
                        elif isinstance(result, dict) and 'response' in result:
                            # Ollama-like response structure
                            response = result["response"]
                        elif isinstance(result, str):
                            response = result
                        else:
                            # Fallback - convert to string
                            response = str(result)
                    except (AttributeError, TypeError, KeyError):
                        # Fallback to generate method for older APIs
                        result = self.llm.generate(
                            messages=[
                                {"role": "system", "content": "You are a helpful RAG assistant that answers questions based on the provided context."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        
                        # Try to extract response from various possible formats
                        if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
                            response = result["choices"][0]["message"]["content"]
                        elif isinstance(result, dict) and 'response' in result:
                            response = result["response"]
                        elif isinstance(result, str):
                            response = result
                        else:
                            # Last resort - convert to string
                            response = str(result)
                except Exception as e:
                    # Fallback if all else fails
                    print(f"Error generating response with LLM: {str(e)}")
                    response = f"I encountered an error while processing your query: {str(e)}"
                
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
            
            try:
                # Try using the Groq LLM
                model_response = self.llm.invoke([
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": followup_prompt}
                ])
                
                # Extract the text based on the response format
                if hasattr(model_response, 'content'):
                    # LangChain standard response
                    followup_text = model_response.content
                elif isinstance(model_response, dict) and 'content' in model_response:
                    followup_text = model_response['content']
                elif isinstance(model_response, dict) and 'choices' in model_response:
                    # Groq-style response
                    choices = model_response.get('choices', [])
                    if choices and 'message' in choices[0]:
                        followup_text = choices[0]['message'].get('content', '')
                    else:
                        followup_text = str(model_response)
                else:
                    followup_text = str(model_response)
                
                # Try to parse the JSON response
                if '```json' in followup_text:
                    # Extract JSON from markdown code block
                    json_str = followup_text.split('```json')[1].split('```')[0].strip()
                elif '```' in followup_text:
                    # Extract from generic code block
                    json_str = followup_text.split('```')[1].strip()
                else:
                    # Use the whole text
                    json_str = followup_text.strip()
                
                # Parse the JSON string to get follow-up questions
                try:
                    parsed_result = json.loads(json_str)
                    if isinstance(parsed_result, list):
                        followup_questions = parsed_result
                    elif isinstance(parsed_result, dict) and 'questions' in parsed_result:
                        followup_questions = parsed_result['questions']
                    else:
                        # If we can't parse as expected, use default questions
                        followup_questions = [
                            "Can you tell me more about this topic?",
                            "What else should I know?",
                            "How does this relate to other topics?"
                        ]
                except json.JSONDecodeError:
                    # Fallback to extracting questions heuristically
                    import re
                    question_pattern = re.compile(r'[^.?!]*\?')
                    followup_questions = question_pattern.findall(followup_text)
                    
                    # If no questions found, use default questions
                    if not followup_questions:
                        followup_questions = [
                            "Can you tell me more about this topic?",
                            "What else should I know?",
                            "How does this relate to other topics?"
                        ]
            except Exception as e:
                logging.warning(f"Error generating follow-up questions with LLM: {e}")
                # Fallback to simple default questions
                followup_questions = [
                    "Can you tell me more about this topic?",
                    "What else should I know?",
                    "How does this relate to other topics?"
                ]
        except Exception as e:
            logging.error(f"Error in follow-up question generation: {e}")
            followup_questions = [
                "Can you tell me more about this topic?",
                "What else should I know?",
                "How does this relate to other topics?"
            ]
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
