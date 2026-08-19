"""
Cross-encoder reranking
"""
from typing import List

import numpy as np
from sentence_transformers import CrossEncoder

from src.logging_config import get_logger
from src.models import RetrievalResult, RerankingResult

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks results using cross-encoder models"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RerankingResult]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Query string
            results: Results to rerank
            top_k: Number of top results to return
        """
        logger.info(f"Reranking {len(results)} results for query: {query}")
        
        if not results:
            return []
        
        # Prepare pairs
        pairs = [[query, result.content] for result in results]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Normalize scores using sigmoid (cross-encoder outputs logits, not probabilities)
        scores = 1 / (1 + np.exp(-scores))
        
        # Create reranked results
        reranked = []
        for original_rank, (result, score) in enumerate(zip(results, scores)):
            reranked_result = RerankingResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                content=result.content,
                relevance_score=float(score),
                original_rank=original_rank,
                new_rank=0,  # Will be set after sorting
            )
            reranked.append(reranked_result)
        
        # Sort by relevance score
        reranked.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update new ranks
        for new_rank, result in enumerate(reranked):
            result.new_rank = new_rank
        
        logger.info(f"Reranking complete, returning top {min(top_k, len(reranked))} results")
        return reranked[:top_k]
