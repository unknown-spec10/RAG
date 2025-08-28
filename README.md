# Gyaan Saarathi

An intelligent document Q&A system powered by RAG (Retrieval-Augmented Generation).

## Features

### Advanced Chunking
- **Semantic Chunking**: Splits documents into coherent chunks using paragraph/topic boundaries and embedding similarity.
- **Recursive Chunking**: Recursively splits documents into hierarchical chunks, with parent/child metadata for multi-level retrieval.
- **Metadata Enrichment**: Each chunk can be enriched with metadata (author, date, section, doc type, etc.) for improved filtering and attribution.
- **Table/Image-Aware Chunking**: Placeholder for extracting tables/images using OCR or table extraction libraries (e.g., pytesseract, camelot).
- **Dynamic Chunking at Query Time**: Supports retrieve-and-expand strategy for very long documents, extracting the most relevant sub-parts for the LLM context.

### Retrieval & Reasoning
- **Hybrid Retrieval**: Combines semantic vector search and keyword-based search, with score blending and optional Jina re-ranking.
- **Context Management**: Dynamically sizes, prioritizes, and condenses context to fit LLM token limits, with protocol refinement for context importance.
- **Confidence Scoring**: The LLM provides a confidence score (0-1) after each answer, indicating how much it relied on the retrieved context versus its own knowledge.
- **Tool Use / Function Calling Ready**: The agent is ready for LangChain-style tool use (e.g., calculator, database, API tools). Just pass your tools to the agent.

### Usage
- Select chunking strategy (`semantic`, `recursive`, `markdown`, etc.) when using `TextChunker`.
- Pass metadata (author, date, doc type, etc.) to enrich chunks.
- For table/image extraction, integrate OCR/table libraries in the provided placeholder.
- Confidence score is returned with every answer for reliability assessment.
- To enable tool use, pass a list of LangChain-compatible tool objects to the agent constructor.

### New Dependencies
- `sentence-transformers` (for semantic chunking and similarity)
- `jina-rerankers` (for document re-ranking)
- `langchain` (for text splitting and tool-use)
- (Optional for future) `pytesseract`, `camelot` (for OCR/table extraction)

---
See code comments and docstrings for more details on each feature.

- PDF document processing and analysis
- Intelligent question answering
- Source citation and page references
- Follow-up question suggestions
- Secure document handling

## Deployment

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

### Production Deployment

1. Set up environment variables in your deployment platform:
   - `GROQ_API_KEY`: Your Groq API key
   - `PYTHON_VERSION`: 3.9.18

2. Deploy to your preferred platform:
   - Heroku: `git push heroku main`
   - AWS: Use Elastic Beanstalk
   - GCP: Use Cloud Run
   - Azure: Use App Service

## Configuration

- Maximum file size: 10MB
- Supported file types: PDF
- Vector store: ChromaDB
- LLM: Groq

## Security

- XSRF protection enabled
- CORS disabled
- Secure file handling
- Environment variable protection

## Monitoring

- Logging enabled
- Prometheus metrics available
- Error tracking configured

## Created and maintained by Deep Podder
