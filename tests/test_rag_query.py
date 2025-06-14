from src.rag.retriever import RAGRetriever
from src.rag.embeddings import TFIDFEmbeddings
from src.agents.rag_agent import RAGAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Any, Optional, Dict, Type

class MockLLM(BaseChatModel):
    """Mock LLM for testing purposes."""
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock"
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response."""
        # Convert the last message to string
        last_message = messages[-1].content if messages else ""
        
        # Generate response based on the message content
        if "follow-up questions" in last_message.lower():
            response_text = "1. What are some specific examples of Python's simplicity?\n2. How does Python compare to other programming languages?\n3. What are the most popular Python frameworks?"
        else:
            response_text = "Based on the context, Python is a versatile programming language known for its simplicity and readability. It supports multiple programming paradigms and has a comprehensive standard library. It's widely used in various fields including data science, machine learning, and web development."
        
        # Create a ChatGeneration with the response
        generation = ChatGeneration(message=AIMessage(content=response_text))
        
        # Return a ChatResult with the generation
        return ChatResult(generations=[generation])

def test_rag_query():
    # Initialize components
    embeddings = TFIDFEmbeddings()
    retriever = RAGRetriever(embeddings=embeddings)
    agent = RAGAgent(retriever=retriever, api_key="dummy_key")
    agent.llm = MockLLM()  # Replace the real LLM with our mock
    
    # Sample documents
    documents = [
        {
            "text": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"source": "python_intro"}
        },
        {
            "text": "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a comprehensive standard library.",
            "metadata": {"source": "python_features"}
        },
        {
            "text": "Python is widely used in data science, machine learning, web development, and automation. Popular frameworks include Django, Flask, and FastAPI for web development.",
            "metadata": {"source": "python_usage"}
        }
    ]
    
    # Add documents to retriever
    retriever.add_documents(documents)
    
    # Test query
    query = "What are the main features of Python?"
    
    # Get response
    result = agent.query(query, documents)
    
    # Print results
    print("\nQuery:", query)
    print("\nResponse:", result["response"])
    print("\nFollow-up Questions:")
    for i, question in enumerate(result["followup_questions"], 1):
        print(f"{i}. {question}")

if __name__ == "__main__":
    test_rag_query() 