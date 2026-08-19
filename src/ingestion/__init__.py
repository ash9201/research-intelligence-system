"""
Ingestion module initialization
"""
from src.ingestion.loader import DocumentLoader
from src.ingestion.parser import DocumentParser, ExtractedPage
from src.ingestion.document import DocumentManager

__all__ = ["DocumentLoader", "DocumentParser", "DocumentManager", "ExtractedPage"]
