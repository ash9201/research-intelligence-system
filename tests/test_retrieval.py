"""
Tests for retrieval system
"""
import pytest
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


class TestBM25Retriever:
    """Test BM25 retriever"""
    
    def test_build_and_retrieve(self, sample_chunks):
        """Test building index and retrieving"""
        bm25 = BM25Retriever()
        bm25.build_index(sample_chunks)
        
        results = bm25.retrieve("machine learning", top_k=2)
        
        assert len(results) > 0
        assert results[0].score >= 0


class TestDenseRetriever:
    """Test dense retriever"""
    
    def test_build_and_retrieve(self, sample_chunks):
        """Test building dense index and retrieving"""
        dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense.build_index(sample_chunks, batch_size=32)
        
        results = dense.retrieve("machine learning", top_k=2)
        
        assert len(results) > 0
        assert 0 <= results[0].score <= 1


class TestHybridRetriever:
    """Test hybrid retriever"""
    
    def test_hybrid_retrieval(self, sample_chunks):
        """Test hybrid retrieval"""
        bm25 = BM25Retriever()
        bm25.build_index(sample_chunks)
        
        dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense.build_index(sample_chunks, batch_size=32)
        
        hybrid = HybridRetriever(bm25, dense, alpha=0.5)
        
        results = hybrid.retrieve("learning", top_k=2)
        
        assert len(results) > 0
        assert all(r.retrieval_method == "hybrid_weighted" for r in results)
        assert all("bm25_normalized_score" in result.metadata for result in results)
        assert all("bm25_raw_score" in result.metadata for result in results)

    def test_rrf_hybrid_retrieval(self, sample_chunks):
        """RRF fusion combines ranks from both sparse and dense result lists."""
        bm25 = BM25Retriever()
        bm25.build_index(sample_chunks)
        dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense.build_index(sample_chunks, batch_size=32)

        hybrid = HybridRetriever(bm25, dense, fusion_strategy="rrf", rrf_k=10)
        results = hybrid.retrieve("learning", top_k=2)

        assert len(results) == 2
        assert all(result.retrieval_method == "hybrid_rrf" for result in results)
        assert all(result.metadata["fusion_strategy"] == "rrf" for result in results)
