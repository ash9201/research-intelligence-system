"""
Document model and management
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib

from src.logging_config import get_logger
from src.models import Document

logger = get_logger(__name__)


class DocumentManager:
    """Manages document lifecycle"""
    
    @staticmethod
    def create_document(
        title: str,
        content: str,
        file_path: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Create a new document"""
        
        # Generate doc_id from file path and content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        doc_id = f"{Path(file_path).stem}_{content_hash}"
        
        doc = Document(
            doc_id=doc_id,
            title=title or Path(file_path).stem,
            content=content,
            file_path=file_path,
            file_type=file_type,
            metadata=metadata or {},
        )
        
        # Add default metadata
        doc.metadata.setdefault("source", file_path)
        doc.metadata.setdefault("char_count", len(content))
        doc.metadata.setdefault("word_count", len(content.split()))
        
        logger.info(f"Created document: {doc.doc_id} from {file_path}")
        return doc
    
    @staticmethod
    def update_document_metadata(
        document: Document,
        metadata: Dict[str, Any],
    ) -> Document:
        """Update document metadata"""
        document.metadata.update(metadata)
        document.updated_at = datetime.utcnow()
        logger.info(f"Updated metadata for document: {document.doc_id}")
        return document
