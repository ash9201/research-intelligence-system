"""
Metadata extraction module
"""
import re
from typing import Any, Dict, List

from src.logging_config import get_logger
from src.models import Chunk

logger = get_logger(__name__)


class MetadataExtractor:
    """Extracts metadata from documents and chunks"""
    
    @staticmethod
    def extract_chunk_metadata(chunk: Chunk) -> Dict[str, Any]:
        """Extract metadata from a chunk"""
        metadata = {
            "char_count": len(chunk.content),
            "word_count": len(chunk.content.split()),
            "sentence_count": len(re.split(r'[.!?]+', chunk.content)),
            "has_code": any(
                marker in chunk.content
                for marker in ["```", "def ", "class ", "function ", "import "]
            ),
            "has_equation": "$$" in chunk.content or "$" in chunk.content,
            "has_list": any(
                chunk.content.startswith(marker)
                for marker in ["- ", "* ", "1. ", "•"]
            ),
        }
        return metadata
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities from text (basic patterns)"""
        entities = {
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "numbers": re.findall(r'\b\d+(?:\.\d+)?\b', text)[:10],  # Limit to first 10
        }
        return entities
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using simple TF-based approach"""
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "should", "could", "may", "might",
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter and count
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:top_k]]
    
    @staticmethod
    def enrich_chunk_metadata(chunk: Chunk) -> Chunk:
        """Enrich a chunk with extracted metadata"""
        # Extract basic metadata
        chunk_metadata = MetadataExtractor.extract_chunk_metadata(chunk)
        chunk.metadata.update(chunk_metadata)
        
        # Extract entities
        entities = MetadataExtractor.extract_entities(chunk.content)
        if any(entities.values()):
            chunk.metadata["entities"] = entities
        
        # Extract keywords
        keywords = MetadataExtractor.extract_keywords(chunk.content)
        if keywords:
            chunk.metadata["keywords"] = keywords
        
        logger.debug(f"Enriched metadata for chunk {chunk.chunk_id}")
        return chunk
