"""
Main entry point for Streamlit Cloud deployment.
This file imports and runs the actual Streamlit application.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual app
from src.frontend.streamlit_app import main

# Run the app
if __name__ == "__main__":
    main()
