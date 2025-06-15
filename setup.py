from setuptools import setup, find_namespace_packages

setup(
    name="rag-system",
    version="0.1.0",
    packages=find_namespace_packages(include=['src*']),
    package_dir={'': '.'},
    install_requires=[
        "streamlit>=1.30.0",
        "pdfplumber>=0.10.0",
        "langchain>=0.1.16",
        "langchain-community>=0.0.32",
        "langgraph>=0.0.20",
        "langchain-text-splitters>=0.0.1",
        "langchain-core>=0.1.46",
        "langsmith>=0.1.17",
        "chromadb>=0.4.22",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "groq>=0.4.1",
        "langchain-groq>=0.1.0",
    ],
    python_requires=">=3.11",
    zip_safe=False,  # Ensure packages are installed as directories
) 