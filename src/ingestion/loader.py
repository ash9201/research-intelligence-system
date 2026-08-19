"""
Document loading and ingestion pipeline
"""
from pathlib import Path
from typing import List, Optional

from src.ingestion.parser import DocumentParser
from src.ingestion.document import DocumentManager
from src.logging_config import get_logger
from src.models import Document

logger = get_logger(__name__)


class DocumentLoader:
    """Loads documents from files"""
    
    @staticmethod
    def load_document(
        file_path: Path,
        metadata: Optional[dict] = None,
    ) -> Document:
        """Load a single document from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Loading document from: {file_path}")
        
        # PDF pages remain available for page-aware chunking; text formats retain their path.
        if file_path.suffix.lower() == ".pdf":
            pages = DocumentParser.parse_pdf_pages(file_path)
            content = "\n".join(f"--- Page {page.number} ---\n{page.text}" for page in pages)
            document_metadata = {
                **(metadata or {}),
                "pages": [{"number": page.number, "text": page.text} for page in pages],
                "page_count": len(pages),
                "pdf_extractor": "pymupdf",
            }
        else:
            content = DocumentParser.parse_document(file_path)
            document_metadata = metadata
        
        # Create document object
        document = DocumentManager.create_document(
            title=file_path.stem,
            content=content,
            file_path=str(file_path),
            file_type=file_path.suffix.lower().lstrip("."),
            metadata=document_metadata,
        )
        
        logger.info(f"Successfully loaded: {document.doc_id}")
        return document
    
    @staticmethod
    def load_documents(
        directory: Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> List[Document]:
        """Load multiple documents from a directory"""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        # Find matching files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                try:
                    doc = DocumentLoader.load_document(file_path)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents


__all__ = ["DocumentLoader", "DocumentParser", "DocumentManager"]
