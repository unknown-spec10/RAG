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
    
    def parse_file(self, file_path: str) -> str:
        """
        Parse a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text from the PDF.
        """
        if self.use_pdfplumber:
            return self._parse_with_pdfplumber(file_path)
        return self._parse_with_pymupdf(file_path)
    
    def parse_cloud_storage(self, bucket_name: str, object_key: str, storage_provider: str = "s3") -> str:
        """
        Parse a PDF file from cloud storage.
        
        Args:
            bucket_name: Name of the cloud storage bucket.
            object_key: Key/path of the PDF file in the bucket.
            storage_provider: Cloud storage provider ('s3' or 'azure').
            
        Returns:
            Extracted text from the PDF.
        """
        if storage_provider == "s3":
            s3 = boto3.client('s3')
            
            try:
                # Download the file to a temporary location
                temp_file = f"/tmp/{object_key.split('/')[-1]}"
                s3.download_file(bucket_name, object_key, temp_file)
                
                # Parse the downloaded file
                text = self.parse_file(temp_file)
                
                # Clean up
                os.remove(temp_file)
                return text
                
            except Exception as e:
                print(f"Error parsing PDF from S3: {str(e)}")
                return ""
        else:
            raise ValueError(f"Unsupported storage provider: {storage_provider}")
    
    def _parse_with_pymupdf(self, file_path: str) -> str:
        """Parse PDF using PyMuPDF."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error parsing PDF with PyMuPDF: {str(e)}")
        return text
    
    def _parse_with_pdfplumber(self, file_path: str) -> str:
        """Parse PDF using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error parsing PDF with pdfplumber: {str(e)}")
        return text
    
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
