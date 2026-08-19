"""
Tests for evaluation metrics
"""
import pytest
from src.evaluation import RetrievalMetrics


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics"""
    
    def test_precision_at_k(self):
        """Test precision@k metric"""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5", "doc6"]
        
        p_at_3 = RetrievalMetrics.precision_at_k(relevant, retrieved, 3)
        assert p_at_3 == 2/3
        
        p_at_5 = RetrievalMetrics.precision_at_k(relevant, retrieved, 5)
        assert p_at_5 == 2/5
    
    def test_recall_at_k(self):
        """Test recall@k metric"""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2"]
        
        r_at_3 = RetrievalMetrics.recall_at_k(relevant, retrieved, 3)
        assert r_at_3 == 2/3
    
    def test_mean_reciprocal_rank(self):
        """Test MRR metric"""
        relevant = ["doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        
        mrr = RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved)
        assert mrr == 1/3
    
    def test_ndcg_at_k(self):
        """Test NDCG@k metric"""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc1", "doc3", "doc2"]
        
        ndcg = RetrievalMetrics.ndcg_at_k(relevant, retrieved, 3)
        assert 0 <= ndcg <= 1
