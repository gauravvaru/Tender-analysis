"""
Handle input documents from the input_docs/ folder
"""
import logging
from pathlib import Path
from typing import List, Dict
import os

from config import INPUT_DOCS_DIR

logger = logging.getLogger(__name__)


class InputDocsHandler:
    """Manage input documents from the input_docs/ folder"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    @staticmethod
    def get_available_documents() -> List[Dict]:
        """
        Get all documents available in input_docs/ folder
        
        Returns:
            List of dicts with file info
        """
        documents = []
        
        try:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(INPUT_DOCS_DIR):
                for file in sorted(files):
                    file_path = Path(root) / file
                    
                    # Check if supported format
                    if file_path.suffix.lower() not in InputDocsHandler.SUPPORTED_EXTENSIONS:
                        continue
                    
                    # Skip hidden files
                    if file.startswith('.'):
                        continue
                    
                    # Get relative path for display
                    relative_path = file_path.relative_to(INPUT_DOCS_DIR)
                    folder_name = str(relative_path.parent) if relative_path.parent != Path('.') else 'input_docs'
                    
                    # Get file size
                    size_bytes = file_path.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                    
                    documents.append({
                        "name": file,
                        "path": str(relative_path),
                        "full_path": str(file_path),
                        "size_mb": round(size_mb, 2),
                        "extension": file_path.suffix.lower(),
                        "folder": folder_name,
                        "size_bytes": size_bytes
                    })
            
            logger.info(f"Found {len(documents)} documents in input_docs/")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    @staticmethod
    def get_documents_by_folder() -> Dict[str, List[Dict]]:
        """Get documents organized by folder"""
        documents = InputDocsHandler.get_available_documents()
        
        organized = {}
        for doc in documents:
            folder = doc["folder"]
            if folder not in organized:
                organized[folder] = []
            organized[folder].append(doc)
        
        return organized
    
    @staticmethod
    def get_total_size_mb() -> float:
        """Get total size of all documents in MB"""
        documents = InputDocsHandler.get_available_documents()
        total_bytes = sum(doc["size_bytes"] for doc in documents)
        return round(total_bytes / (1024 * 1024), 2)