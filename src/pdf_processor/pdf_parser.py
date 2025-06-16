from typing import Dict, Any

class PDFParser:
    def get_support_info(self, page_number: int) -> Dict[str, Any]:
        """Get support information for a specific page."""
        try:
            # Get the page
            page = self.document[page_number - 1]
            
            # Extract text from the page
            text = page.get_text()
            
            # Extract support information
            support_info = self.extract_support_info(text)
            
            return support_info
        except Exception as e:
            logger.error(f"Error getting support info for page {page_number}: {str(e)}")
            return {} 