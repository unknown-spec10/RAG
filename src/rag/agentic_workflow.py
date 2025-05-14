"""
Agentic RAG workflow integration for static and user-specific KBs.
"""
import os
from typing import Optional, List, Dict, Any
from src.pdf_processor.pdf_parser import PDFParser
from src.pdf_processor.text_chunker import TextChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vector_db.faiss_db import FAISSVectorDB
from src.rag.retriever import RAGRetriever

import requests
from bs4 import BeautifulSoup
import tempfile
import uuid

# Mapping: insurer name → {doc_type: url}
INSURER_URLS = {
    "HDFC Ergo": {
        "Brochure": "https://www.hdfcergo.com/health-insurance/optima-secure#brochure",
        "Policy Wordings": "https://www.hdfcergo.com/health-insurance/optima-secure#policy-wordings",
        "Benefits Guide": "https://www.hdfcergo.com/health-insurance/optima-secure#benefit-guide"
    },
    "Niva Bupa": {
        "Brochure": "https://www.nivabupa.com/health-insurance-plans.html",
        "Policy Wordings": "https://www.nivabupa.com/customer-service/download-policy-wordings.html",
        "Benefits Guide": "https://www.nivabupa.com/customer-service/download-benefit-guide.html"
    },
    "Star Health": {
        "Brochure": "https://www.starhealth.in/health-insurance-policy-brochures",
        "Policy Wordings": "https://www.starhealth.in/health-insurance-policy-wordings",
        "Benefits Guide": "https://www.starhealth.in/health-insurance-benefit-guide"
    },
    "ICICI Lombard": {
        "Brochure": "https://www.icicilombard.com/health-insurance/downloads/brochures",
        "Policy Wordings": "https://www.icicilombard.com/health-insurance/downloads/policy-wordings",
        "Benefits Guide": "https://www.icicilombard.com/health-insurance/downloads/benefit-guide"
    },
    "Tata AIG": {
        "Brochure": "https://www.tataaig.com/downloads",
        "Policy Wordings": "https://www.tataaig.com/downloads",
        "Benefits Guide": "https://www.tataaig.com/downloads"
    }
}

def fetch_pdf_links(url):
    """
    Fetch all PDF links from a given page using requests+BeautifulSoup.
    """
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                # Handle relative URLs
                if not href.startswith("http"):
                    href = requests.compat.urljoin(url, href)
                pdf_links.append(href)
        return pdf_links
    except Exception as e:
        print(f"Error fetching PDF links from {url}: {e}")
        return []

def fetch_pdf_links_selenium(url):
    """
    Fetch all PDF links from a given page using Selenium (for dynamic JS sites).
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    pdf_links = []
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        # Wait for JS to load (can be improved with explicit waits)
        driver.implicitly_wait(5)
        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            href = a.get_attribute("href")
            if href and href.lower().endswith(".pdf"):
                pdf_links.append(href)
        driver.quit()
    except Exception as e:
        print(f"Selenium error fetching PDF links from {url}: {e}")
    return pdf_links

def download_pdf(url, dest_dir):
    """
    Download a PDF from url to dest_dir, returns local file path.
    """
    local_filename = os.path.join(dest_dir, url.split("/")[-1])
    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def scrape_and_ingest_pdfs_from_page(page_url):
    """
    Scrape a page for PDF links, download and ingest each PDF into static KB.
    Tries requests+bs4 first, then Selenium if no PDFs found.
    """
    pdf_links = fetch_pdf_links(page_url)
    if not pdf_links:
        print("No PDFs found with requests+bs4, trying Selenium...")
        pdf_links = fetch_pdf_links_selenium(page_url)
    print(f"Found {len(pdf_links)} PDF links on {page_url}")
    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for pdf_url in pdf_links:
            try:
                local_pdf = download_pdf(pdf_url, tmpdir)
                print(f"Ingesting {local_pdf}")
                ingest_document("static", local_pdf)
                count += 1
            except Exception as e:
                print(f"Failed to ingest {pdf_url}: {e}")
    return count

STATIC_KB_DIR = os.path.join("data", "static_faiss")
USER_KB_DIR = os.path.join("data", "user_faiss")
STATIC_COLLECTION = "static_kb"
USER_COLLECTION = "user_kb"

EMBED_DIM = 384  # Matches all-MiniLM-L6-v2 and FAISS config

def ingest_document(mode: str, file_path: str, user_id: Optional[str] = None):
    """
    Ingest a PDF document into either the static or user-specific KB.
    Args:
        mode: 'static' or 'user'
        file_path: path to PDF file
        user_id: required if mode == 'user'
    """
    parser = PDFParser()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    embedder = EmbeddingModel(embedding_dim=EMBED_DIM)

    text = parser.parse_file(file_path)
    chunks = chunker.chunk_text(text)
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.encode(chunk_texts)

    if mode == 'static':
        db = FAISSVectorDB(STATIC_KB_DIR, STATIC_COLLECTION, dimension=EMBED_DIM)
        metadatas = [{"source": os.path.basename(file_path)} for _ in chunks]
    elif mode == 'user':
        if not user_id:
            raise ValueError("user_id is required for user-specific ingestion.")
        db = FAISSVectorDB(USER_KB_DIR, USER_COLLECTION, dimension=EMBED_DIM)
        metadatas = [{"user_id": user_id, "source": os.path.basename(file_path)} for _ in chunks]
    else:
        raise ValueError("mode must be 'static' or 'user'")

    db.add_documents(chunk_texts, embeddings, metadatas)
    return len(chunks)

def retrieve_context(query: str, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve context from both static and user-specific KBs for a query.
    Args:
        query: user query
        user_id: id of the user
        top_k: number of docs from each KB
    Returns:
        List of context dicts (static + user-specific)
    """
    embedder = EmbeddingModel(embedding_dim=EMBED_DIM)
    # Static KB
    static_db = FAISSVectorDB(STATIC_KB_DIR, STATIC_COLLECTION, dimension=EMBED_DIM)
    static_retriever = RAGRetriever(static_db, embedder, top_k=top_k)
    static_results = static_retriever.retrieve(query)
    # User KB
    user_db = FAISSVectorDB(USER_KB_DIR, USER_COLLECTION, dimension=EMBED_DIM)
    user_retriever = RAGRetriever(user_db, embedder, top_k=top_k)
    user_results = user_retriever.retrieve(query, filter_criteria={"user_id": user_id})
    # Combine
    return (user_results or []) + (static_results or [])
