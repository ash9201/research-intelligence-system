"""Hybrid retrieval with explicit score-normalized and rank-based fusion."""
from typing import Dict, List

from src.logging_config import get_logger
from src.models import RetrievalResult
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining sparse (BM25) and dense methods"""
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        fusion_strategy: str = "weighted",
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            bm25_retriever: BM25 retriever instance
            dense_retriever: Dense retriever instance
            alpha: Weight for dense results (1-alpha for BM25)
            fusion_strategy: "weighted" uses max-normalized scores; "rrf" uses ranks
            rrf_k: Reciprocal Rank Fusion smoothing constant
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if fusion_strategy not in {"weighted", "rrf"}:
            raise ValueError("fusion_strategy must be 'weighted' or 'rrf'")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_k: int = 20,
        dense_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid method
        
        Args:
            query: Query string
            top_k: Number of final results
            bm25_k: Number of BM25 results to retrieve
            dense_k: Number of dense results to retrieve
        """
        logger.info(f"Hybrid retrieval for query: {query}")
        
        # Get results from both methods
        bm25_results = self.bm25.retrieve(query, top_k=bm25_k)
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        
        if self.fusion_strategy == "rrf":
            final_results = self._fuse_rrf(bm25_results, dense_results)
        else:
            final_results = self._fuse_weighted(bm25_results, dense_results)
        
        # Sort by hybrid score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k
        return final_results[:top_k]

    @staticmethod
    def _normalized_scores(results: List[RetrievalResult]) -> Dict[str, float]:
        """Max-normalize one retrieval stream; raw scores remain in metadata."""
        maximum = max((result.score for result in results), default=0.0)
        if maximum <= 0:
            return {result.chunk_id: 0.0 for result in results}
        return {result.chunk_id: result.score / maximum for result in results}

    def _fuse_weighted(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Fuse explicit max-normalized BM25 and dense scores using alpha."""
        bm25_raw = {result.chunk_id: result.score for result in bm25_results}
        dense_raw = {result.chunk_id: result.score for result in dense_results}
        bm25_normalized = self._normalized_scores(bm25_results)
        dense_normalized = self._normalized_scores(dense_results)
        candidates = {result.chunk_id: result for result in bm25_results + dense_results}
        final_results = []
        for chunk_id, result in candidates.items():
            bm25_score = bm25_normalized.get(chunk_id, 0.0)
            dense_score = dense_normalized.get(chunk_id, 0.0)
            fused_score = (1 - self.alpha) * bm25_score + self.alpha * dense_score
            final_results.append(
                result.model_copy(
                    update={
                        "score": fused_score,
                        "retrieval_method": "hybrid_weighted",
                        "metadata": {
                            **result.metadata,
                            "fusion_strategy": "weighted",
                            "bm25_raw_score": bm25_raw.get(chunk_id),
                            "dense_raw_score": dense_raw.get(chunk_id),
                            "bm25_normalized_score": bm25_score,
                            "dense_normalized_score": dense_score,
                        },
                    }
                )
            )
        return final_results

    def _fuse_rrf(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Fuse result ranks using $1 / (rrf_k + rank)$ per retrieval stream."""
        candidates = {result.chunk_id: result for result in bm25_results + dense_results}
        bm25_ranks = {result.chunk_id: rank for rank, result in enumerate(bm25_results, start=1)}
        dense_ranks = {result.chunk_id: rank for rank, result in enumerate(dense_results, start=1)}
        final_results = []
        for chunk_id, result in candidates.items():
            bm25_contribution = 1 / (self.rrf_k + bm25_ranks[chunk_id]) if chunk_id in bm25_ranks else 0.0
            dense_contribution = 1 / (self.rrf_k + dense_ranks[chunk_id]) if chunk_id in dense_ranks else 0.0
            final_results.append(
                result.model_copy(
                    update={
                        "score": bm25_contribution + dense_contribution,
                        "retrieval_method": "hybrid_rrf",
                        "metadata": {
                            **result.metadata,
                            "fusion_strategy": "rrf",
                            "rrf_k": self.rrf_k,
                            "bm25_rank": bm25_ranks.get(chunk_id),
                            "dense_rank": dense_ranks.get(chunk_id),
                        },
                    }
                )
            )
        return final_results
