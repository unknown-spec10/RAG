"""Test script for Groq API connection."""
import os
from langchain_groq import ChatGroq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_groq_connection():
    """Test the Groq API connection."""
    try:
        # Get API key from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("No API key found in environment variables")
            return False

        # Print masked API key for debugging
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        logger.info(f"Using API key: {masked_key}")

        # Initialize Groq client
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-70b-8192",
            temperature=0.1,
            max_tokens=100
        )

        # Test with a simple prompt
        response = llm.invoke("Say 'Hello, World!'")
        
        # Print the response
        logger.info("API Test Response:")
        logger.info(response.content)
        
        return True

    except Exception as e:
        logger.error(f"Error testing Groq API: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Testing Groq API connection...")
    success = test_groq_connection()
    if success:
        logger.info("✅ Groq API test successful!")
    else:
        logger.error("❌ Groq API test failed!") 