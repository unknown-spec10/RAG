"""PDF Parser module for extracting text from PDF files."""
import pdfplumber
from typing import List, Dict, Any, Optional
import os


class PDFParser:
    """Class for parsing PDF documents."""
    
    def __init__(self):
        """Initialize the PDF parser."""
        pass
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and extract text with page numbers.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of dictionaries containing text and page numbers.
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                documents = []
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:  # Only add pages with text
                        documents.append({
                            "text": text,
                            "metadata": {
                                "page": page_num,
                                "total_pages": len(pdf.pages)
                            }
                        })
                return documents
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")
    
    def parse_cloud_storage(self, bucket_name: str, object_key: str, storage_provider: str = "s3") -> List[Dict[str, Any]]:
        """
        Parse a PDF file from cloud storage.
        
        Args:
            bucket_name: Name of the cloud storage bucket.
            object_key: Key/path of the PDF file in the bucket.
            storage_provider: Cloud storage provider ('s3' or 'azure').
            
        Returns:
            List of dictionaries containing text and page numbers.
        """
        raise NotImplementedError("Cloud storage parsing is not supported in this implementation")
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            A dictionary containing metadata about the PDF.
        """
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata
            # Convert metadata to a more readable format
            return {
                "title": metadata.get('Title', ''),
                "author": metadata.get('Author', ''),
                "creator": metadata.get('Creator', ''),
                "producer": metadata.get('Producer', ''),
                "creation_date": metadata.get('CreationDate', ''),
                "mod_date": metadata.get('ModDate', ''),
                "total_pages": len(pdf.pages)
            }
