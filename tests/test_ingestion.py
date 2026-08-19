"""
Tests for document ingestion
"""
import pytest
from pathlib import Path
from src.ingestion import DocumentLoader, DocumentParser


class TestDocumentParser:
    """Test document parsing"""
    
    def test_parse_text_document(self, sample_text_document):
        """Test parsing a text document"""
        content = DocumentParser.parse_text(sample_text_document)
        assert content is not None
        assert len(content) > 0
        assert "Machine learning" in content
    
    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file"""
        with pytest.raises(FileNotFoundError):
            DocumentParser.parse_text(Path("nonexistent.txt"))

    def test_pdf_normalization_preserves_technical_symbols_and_word_boundaries(self):
        """Layout repair must not invent LaTex or silently discard formula characters."""
        raw = "The model-\ndimensional value is √dₖ.\nThe first warmup_steps are used."

        normalized = DocumentParser.normalize_pdf_text(raw)

        assert "model-dimensional" in normalized
        assert "√dₖ" in normalized
        assert "warmup_steps" in normalized
        assert "\\sqrt" not in normalized
        assert "modeldimensional" not in normalized


class TestDocumentLoader:
    """Test document loading"""
    
    def test_load_single_document(self, sample_text_document):
        """Test loading a single document"""
        doc = DocumentLoader.load_document(sample_text_document)
        assert doc is not None
        assert doc.doc_id is not None
        assert len(doc.content) > 0
    
    def test_load_documents_from_directory(self, test_data_dir, sample_text_document):
        """Test loading documents from directory"""
        docs = DocumentLoader.load_documents(test_data_dir, pattern="*.txt")
        assert len(docs) > 0
    
    def test_document_metadata(self, sample_text_document):
        """Test document metadata extraction"""
        doc = DocumentLoader.load_document(sample_text_document)
        assert doc.metadata.get("char_count") == len(doc.content)
        assert doc.metadata.get("word_count") > 0
