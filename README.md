# Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent document Q&A using advanced language models. This system combines document processing, semantic search, and LLM capabilities to provide accurate, context-aware responses.

## 🚀 Features

- **Document Processing**
  - PDF parsing and text extraction
  - Intelligent text chunking with overlap
  - Metadata preservation (source, page numbers)

- **Semantic Search**
  - TF-IDF based document retrieval
  - Context-aware document ranking
  - Efficient document chunking

- **LLM Integration**
  - Groq API integration for high-performance inference
  - Context-aware response generation
  - Source attribution and verification

- **User Interface**
  - Streamlit-based interactive interface
  - Document upload and processing
  - Real-time Q&A with source citations
  - Expandable source references

## 🏗️ System Architecture

### Core Components

1. **PDF Processor**
   - `PDFParser`: Extracts text from PDF documents
   - `TextChunker`: Splits documents into semantic chunks
   - Preserves document structure and metadata

2. **RAG Components**
   - `RAGRetriever`: Manages document retrieval
   - `TFIDFEmbeddings`: Generates document embeddings
   - `ContextProtocolManager`: Handles context management

3. **Agent System**
   - `RAGAgent`: Orchestrates the RAG workflow
   - `MockLLM`: Fallback for testing and error cases
   - LangGraph integration for workflow management

4. **User Interface**
   - Streamlit app for user interaction
   - Document upload and management
   - Interactive Q&A interface

## 🔄 Workflow

1. **Document Processing**
   ```
   PDF Upload → Text Extraction → Chunking → Embedding Generation
   ```

2. **Query Processing**
   ```
   User Query → Semantic Search → Context Retrieval → Response Generation
   ```

3. **Response Generation**
   ```
   Context + Query → LLM Processing → Source Attribution → Response
   ```

## 🛠️ Technical Implementation

### Document Processing
```python
# PDF parsing and chunking
pdf_parser = PDFParser()
text = pdf_parser.parse_file(file_path)
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_documents([{"text": text, "metadata": {"source": filename}}])
```

### RAG Implementation
```python
# RAG agent initialization
embeddings = TFIDFEmbeddings()
retriever = RAGRetriever(embeddings=embeddings)
agent = RAGAgent(retriever=retriever, api_key=api_key)
```

### Response Generation
```python
# Query processing
result = agent.query(query, documents)
response = result["response"]
sources = result["sources"]
```

## 📦 Dependencies

- **Core Dependencies**
  - `streamlit==1.30.0`: Web interface
  - `langchain==0.1.0`: RAG framework
  - `langchain-groq==0.1.0`: Groq integration
  - `pdfplumber==0.10.0`: PDF processing

- **Vector Store & Embeddings**
  - `faiss-cpu==1.7.4`: Vector similarity search
  - `scikit-learn==1.3.0`: TF-IDF implementation

## 🚀 Getting Started

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Create `.streamlit/secrets.toml`:
     ```toml
     groq_api_key = "your-groq-api-key"
     ```

3. **Running the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔍 How It Works

1. **Document Processing**
   - PDFs are uploaded and processed
   - Text is extracted and chunked
   - Chunks are embedded and stored

2. **Query Handling**
   - User submits a question
   - System retrieves relevant chunks
   - Context is prepared for the LLM

3. **Response Generation**
   - LLM generates response using context
   - Sources are tracked and attributed
   - Response is formatted with citations

4. **Source Attribution**
   - Each response includes source references
   - Sources are expandable for verification
   - Page numbers are included when available

## 🎯 Key Features

- **Context-Aware Responses**
  - Only uses provided document context
  - Clear indication when information isn't available
  - No hallucination or external knowledge

- **Source Tracking**
  - Detailed source attribution
  - Expandable source references
  - Page number tracking

- **Error Handling**
  - Graceful fallback to mock LLM
  - Clear error messages
  - Robust document processing

## 🔒 Security & Privacy

- API keys stored in Streamlit secrets
- No data persistence between sessions
- Secure document processing

## 🛠️ Development

### Adding New Features

1. **New Document Types**
   - Extend `PDFParser` class
   - Implement new parser methods
   - Update chunking strategy

2. **Custom Embeddings**
   - Implement new embedding class
   - Update retriever configuration
   - Test with different models

3. **UI Enhancements**
   - Modify Streamlit components
   - Add new visualization features
   - Enhance user experience

## 📝 Best Practices

1. **Document Processing**
   - Use appropriate chunk sizes
   - Maintain metadata
   - Handle errors gracefully

2. **Response Generation**
   - Always cite sources
   - Be explicit about limitations
   - Provide clear follow-up questions

3. **Error Handling**
   - Implement proper logging
   - Provide user feedback
   - Maintain system stability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License

## 👥 Authors

- Deep Podder
  - GitHub: [unknown-spec10](https://github.com/unknown-spec10)
  - LinkedIn: [deeppodder2005](http://www.linkedin.com/in/deeppodder2005)

## Streamlit Cloud Deployment

To deploy this application to Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file path (`streamlit_app.py`)
6. Add your secrets:
   - Go to the app's settings
   - Add the following secret:
     - `groq_api_key`: Your Groq API key (should start with `gsk_`)

The app will be automatically deployed and you'll get a public URL to share.

### Important Notes for Deployment

- The app is configured to handle PDF files up to 200MB
- Make sure your Groq API key is properly set in Streamlit Cloud secrets
- The app uses TFIDF embeddings which are lightweight and suitable for cloud deployment
- All dependencies are pinned to specific versions for stability
