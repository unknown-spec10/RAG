# Agentic RAG

A Retrieval-Augmented Generation (RAG) application that allows you to chat with your documents using advanced LLMs via the Groq API. Created by Deep Podder.

## Features

- Upload PDF documents for processing
- Chat with your documents using state-of-the-art language models
- View sources for generated responses
- Clean and intuitive user interface

## Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Create a new app on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect to your forked repository
4. Configure the app settings with the main file as `streamlit_app.py`
5. Add your API keys as secrets in the Streamlit Cloud dashboard:
   - Go to Advanced Settings > Secrets
   - Add the following to your secrets.toml:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   GOOGLE_API_KEY = "your-google-api-key-here"  # For Gemini embeddings (optional)
   ```

### Troubleshooting Cloud Deployment

- **Memory Issues**: The application uses lightweight embedding methods to stay within Streamlit Cloud's memory limits
- **Missing API Keys**: Check that you've properly added your API keys to the Streamlit secrets
- **Slow Performance**: First document processing may take longer; subsequent queries will be faster
- **Embedding Errors**: If Google API isn't available, the app will automatically fall back to hash-based embeddings

## Local Development

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up your API keys using one of these methods:
   
   **Option 1**: Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your-groq-api-key-here
   GOOGLE_API_KEY=your-google-api-key-here  # For Gemini embeddings (optional)
   ```
   
   **Option 2**: Create a `.streamlit/secrets.toml` file:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   GOOGLE_API_KEY = "your-google-api-key-here"  # For Gemini embeddings (optional)
   ```

6. Run the application: `streamlit run streamlit_app.py`

## Requirements

- Python 3.9+
- Groq API key (required) - Get one from [Groq Console](https://console.groq.com/)
- Google API key for Gemini embeddings (optional, falls back to hash-based embeddings if not provided)

## Features & Limitations

### Features
- **Document Processing**: Upload and process PDFs into searchable chunks
- **Retrieval**: Find relevant document sections based on your questions
- **Generation**: Get AI responses based on your document content using Groq's Llama3-70b model
- **Source Attribution**: See which parts of your documents were used to generate the answer

### Limitations
- PDF processing only (no Word, Excel or other formats)
- Maximum file size limited by Streamlit (default 200MB)
- Performance depends on document complexity and size

## Credits

Created and maintained by Deep Podder.

---

Created and maintained by Deep Podder
[GitHub](https://github.com/unknown-spec10) | [LinkedIn](http://www.linkedin.com/in/deeppodder2005)

A powerful agentic RAG (Retrieval Augmented Generation) application that allows you to add PDF documents to your knowledge base and ask questions based on their content.

## Features

- **PDF Processing**: Upload and process PDF documents using PyMuPDF and pdfplumber
- **Intelligent Chunking**: Break down documents into semantic chunks for better retrieval
- **Advanced Embeddings**: Generate embeddings using sentence-transformers (all-MiniLM or BGE)
- **Vector Database**: Store and retrieve document chunks efficiently using ChromaDB
- **Agentic RAG**: Use LangGraph for orchestrating the RAG workflow
- **Context Protocol**: Implement Model Context Protocol for standardized context handling
- **Local LLM Integration**: Connect to Ollama for local LLM inference (llama3, mistral)
- **Interactive UI**: User-friendly interface built with Streamlit

## Project Structure

```
Agentic_Rag/
├── app.py                  # Main entry point
├── README.md               # This file
├── requirements.txt        # Dependencies
├── data/                   # Data directory
│   ├── pdfs/               # Stored PDF files
│   └── chroma_db/          # ChromaDB storage
└── src/                    # Source code
    ├── pdf_processor/      # PDF parsing and chunking
    ├── embeddings/         # Embedding models
    ├── vector_db/          # Vector database
    ├── rag/                # RAG components
    ├── agents/             # Agentic components
    ├── frontend/           # UI components
    └── utils/              # Utility functions
```

## Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd Agentic_Rag
   ```

2. **Set up a virtual environment**:
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```
     .\venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

5. **Install Ollama** (for local LLM inference):
   - Follow instructions at [ollama.ai](https://ollama.ai/)
   - Pull the required models:
     ```
     ollama pull llama3
     ```

## Usage

### Running the Application Locally

Start the Streamlit application:

```
python -m streamlit run streamlit_app.py
```

Or simply:

```
streamlit run streamlit_app.py
```

### Using the Application

1. **Upload Documents**:
   - Go to the "Upload & Manage" tab
   - Upload PDF files
   - Configure chunking parameters and click "Index Documents"

2. **Ask Questions**:
   - Go to the "Chat" tab
   - Type your question in the input box
   - View the response and sources
   - Try suggested follow-up questions

## Deploying to Streamlit Cloud

This application is designed to work seamlessly with Streamlit Cloud. Follow these steps to deploy:

1. **Push your code to GitHub**:
   - Create a new GitHub repository
   - Push your local code to the repository

2. **Set up environment variables**:
   - In Streamlit Cloud, go to your app settings
   - Add the following secrets:
     - `OPENAI_API_KEY`: Your OpenAI API key for model access

3. **Deploy the app**:
   - Connect your GitHub repository to Streamlit Cloud
   - Select the repository and branch to deploy
   - Set the main file path to `streamlit_app.py`
   - Click 'Deploy'

### Important Notes for Cloud Deployment

- The app will automatically use OpenAI's models when deployed to Streamlit Cloud
- Local files won't persist between sessions unless you use Streamlit Cloud's file storage APIs
- Indexed documents will be stored in the app's session state while the session is active

### Configuration

You can configure the application using environment variables or by creating a `.env` file:

```
# LLM Settings
AGENTIC_RAG_LLM_MODEL_NAME=llama3
AGENTIC_RAG_LLM_TEMPERATURE=0.1

# Embedding Settings
AGENTIC_RAG_EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2

# Retriever Settings
AGENTIC_RAG_RETRIEVER_TOP_K=5
```

## Advanced Usage

### Programmatic API

You can use the components programmatically:

```python
from app import setup_components

# Set up components
pdf_parser, text_chunker, embedding_model, vector_db, retriever, agent = setup_components()

# Process a PDF
documents = pdf_parser.parse_pdf("path/to/document.pdf")
chunked_docs = text_chunker.chunk_documents(documents)
embedded_docs = embedding_model.embed_documents(chunked_docs)
vector_db.add_documents(embedded_docs)

# Query the knowledge base
result = agent.query("What is the main topic of the document?")
print(result["response"])
```

## Development

### Adding New Features

1. **New Document Types**:
   - Extend the document processing in `src/pdf_processor/`

2. **New Embedding Models**:
   - Add new models in `src/embeddings/embedding_model.py`

3. **Custom Agents**:
   - Create new agent types in `src/agents/`

## License

MIT

## Acknowledgements

- LangChain and LangGraph for the RAG and orchestration frameworks
- Sentence Transformers for embedding models
- ChromaDB for vector storage
- Ollama for local LLM integration
- Streamlit for the user interface
