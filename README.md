# Gyaan Saarathi

An intelligent document Q&A system powered by RAG (Retrieval-Augmented Generation).

## Features

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
