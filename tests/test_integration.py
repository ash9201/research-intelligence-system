"""
Integration tests for the full pipeline
"""
import pytest
import tempfile
from pathlib import Path
from src.ingestion import DocumentLoader
from src.chunking import DocumentChunker
from src.metadata import MetadataExtractor
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker


@pytest.mark.integration
class TestFullPipeline:
    """Test the complete retrieval pipeline"""
    
    def test_end_to_end_retrieval(self, sample_text_document):
        """Test end-to-end retrieval pipeline"""
        # Load document
        doc = DocumentLoader.load_document(sample_text_document)
        assert doc is not None
        
        # Chunk document
        chunker = DocumentChunker(chunk_size=256, overlap=64)
        chunks = chunker.chunk_document(doc.content, doc.doc_id)
        assert len(chunks) > 0
        
        # Enrich metadata
        chunks = [MetadataExtractor.enrich_chunk_metadata(chunk) for chunk in chunks]
        
        # Build index
        with tempfile.TemporaryDirectory() as tmpdir:
            index_manager = IndexManager(Path(tmpdir))
            retriever = index_manager.create_index("test_index", chunks)
            
            # Retrieve
            results = retriever.retrieve("machine learning", top_k=3)
            
            assert len(results) > 0
            assert all(0 <= r.score <= 1 for r in results)
    
    def test_reranking_integration(self, sample_text_document):
        """Test reranking in the pipeline"""
        # Load and chunk
        doc = DocumentLoader.load_document(sample_text_document)
        chunker = DocumentChunker(chunk_size=256, overlap=64)
        chunks = chunker.chunk_document(doc.content, doc.doc_id)
        
        # Build index and retrieve
        with tempfile.TemporaryDirectory() as tmpdir:
            index_manager = IndexManager(Path(tmpdir))
            retriever = index_manager.create_index("test_index", chunks)
            
            # Retrieve
            results = retriever.retrieve("machine learning", top_k=5)
            
            # Rerank
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank("machine learning", results, top_k=3)
            
            assert len(reranked) <= 3
            assert all(0 <= r.relevance_score <= 1 for r in reranked)
