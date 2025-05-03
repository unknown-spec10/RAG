"""
Agentic RAG Application - Main Entry Point
-----------------------------------------

This script runs the main application. You can either:
1. Run the Streamlit web interface using: streamlit run app.py
2. Import the components directly for programmatic use
"""
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Run the Streamlit application."""
    import streamlit.web.bootstrap as bootstrap
    
    # Ensure data directories exist
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/chroma_db", exist_ok=True)
    
    # Run Streamlit app
    filename = os.path.join(project_root, "src/frontend/streamlit_app.py")
    bootstrap.run(filename, "", [], [])

if __name__ == "__main__":
    main()
