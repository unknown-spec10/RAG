from setuptools import setup, find_packages

setup(
    name="gyaan_saarathi",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.30.0",
        "python-dotenv==1.0.0",
        "pydantic==2.10.3",
        "pdfplumber==0.10.0",
        "langchain==0.1.16",
        "langchain-community>=0.0.32,<0.1",
        "langgraph==0.0.20",
        "langchain-text-splitters==0.0.1",
        "langchain-core==0.1.46",
        "chromadb==0.4.22",
        "numpy>=1.26.0",
        "scikit-learn==1.3.0",
        "groq==0.4.1",
        "langchain-groq==0.1.0"
    ],
    python_requires=">=3.9",
) 