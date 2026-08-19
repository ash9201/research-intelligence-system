"""
Evaluation metrics for retrieval
"""
from typing import List

from src.logging_config import get_logger
from src.models import RetrievalResult

logger = get_logger(__name__)


class RetrievalMetrics:
    """Compute retrieval evaluation metrics"""
    
    @staticmethod
    def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Precision@k metric
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Cutoff position
        """
        if k <= 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_at_k = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
        
        return relevant_at_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Recall@k metric
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Cutoff position
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_at_k = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_ids)
        
        return relevant_at_k / len(relevant_ids)
    
    @staticmethod
    def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR) metric
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@k)
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Cutoff position
        """
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_ids:
                dcg += 1.0 / (1.0 + (rank - 1))
        
        # IDCG (ideal DCG)
        idcg = 0.0
        for rank in range(1, min(len(relevant_ids), k) + 1):
            idcg += 1.0 / (1.0 + (rank - 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Hit rate @ k (is there at least one relevant document in top-k?)
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Cutoff position
        """
        retrieved_at_k = set(retrieved_ids[:k])
        for doc_id in relevant_ids:
            if doc_id in retrieved_at_k:
                return 1.0
        
        return 0.0


class EvaluationFramework:
    """Unified evaluation framework"""
    
    @staticmethod
    def evaluate_retrieval(
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k_values: List[int] = [1, 5, 10],
    ) -> dict:
        """
        Evaluate retrieval results
        
        Args:
            relevant_ids: List of relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k_values: List of k values for metrics
        """
        results = {
            "mrr": RetrievalMetrics.mean_reciprocal_rank(relevant_ids, retrieved_ids),
        }
        
        for k in k_values:
            results[f"precision@{k}"] = RetrievalMetrics.precision_at_k(
                relevant_ids, retrieved_ids, k
            )
            results[f"recall@{k}"] = RetrievalMetrics.recall_at_k(
                relevant_ids, retrieved_ids, k
            )
            results[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(
                relevant_ids, retrieved_ids, k
            )
            results[f"hit_rate@{k}"] = RetrievalMetrics.hit_rate_at_k(
                relevant_ids, retrieved_ids, k
            )
        
        return results
