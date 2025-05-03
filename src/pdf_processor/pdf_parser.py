"""PDF Parser module for extracting text from PDF files."""
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any, Optional
import os


class PDFParser:
    """Class for parsing PDF documents."""
    
    def __init__(self, use_pdfplumber: bool = False):
        """
        Initialize the PDF parser.
        
        Args:
            use_pdfplumber: Whether to use pdfplumber instead of PyMuPDF.
        """
        self.use_pdfplumber = use_pdfplumber
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            A list of dictionaries, each containing the page number and text.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.use_pdfplumber:
            return self._parse_with_pdfplumber(pdf_path)
        else:
            return self._parse_with_pymupdf(pdf_path)
    
    def _parse_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Parse PDF with PyMuPDF."""
        result = []
        
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                result.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "metadata": {
                        "file_path": pdf_path,
                        "page_count": len(doc),
                    }
                })
            doc.close()
        except Exception as e:
            raise Exception(f"Error parsing PDF with PyMuPDF: {str(e)}")
        
        return result
    
    def _parse_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Parse PDF with pdfplumber."""
        result = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    result.append({
                        "page_num": page_num + 1,
                        "text": text,
                        "metadata": {
                            "file_path": pdf_path,
                            "page_count": len(pdf.pages),
                        }
                    })
        except Exception as e:
            raise Exception(f"Error parsing PDF with pdfplumber: {str(e)}")
        
        return result
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            A dictionary containing metadata about the PDF.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "page_count": len(doc),
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
            }
            doc.close()
            return metadata
        except Exception as e:
            raise Exception(f"Error extracting metadata: {str(e)}")
