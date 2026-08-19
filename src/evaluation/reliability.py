"""
Confidence and reliability estimation
"""
from typing import Dict, List, Optional

from src.logging_config import get_logger
from src.models import AnswerReliability, Citation, RerankingResult

logger = get_logger(__name__)


class ReliabilityEstimator:
    """Estimates confidence and reliability of answers"""
    
    @staticmethod
    def estimate_answer_confidence(
        answer: str,
        citations: List[Citation],
        num_sources: int,
    ) -> float:
        """
        Estimate confidence in generated answer
        
        Args:
            answer: Generated answer text
            citations: Supporting citations
            num_sources: Number of sources used
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Citation count (up to 0.3 points)
        citation_factor = min(len(citations) / 3.0, 0.3)
        confidence += citation_factor
        
        # Factor 2: Source diversity (up to 0.1 points)
        unique_docs = len(set(c.doc_id for c in citations))
        diversity_factor = min(unique_docs / 3.0, 0.1)
        confidence += diversity_factor
        
        # Factor 3: Average citation quality (up to 0.1 points)
        if citations:
            avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
            quality_factor = avg_relevance * 0.1
            confidence += quality_factor
        
        # Factor 4: Answer length (up to 0.1 points)
        answer_length = len(answer.split())
        length_factor = min(answer_length / 100.0, 0.1)
        confidence += length_factor
        
        return min(confidence, 1.0)
    
    @staticmethod
    def assess_retrieval_confidence(
        top_k_results: List[RerankingResult],
    ) -> Dict[str, float]:
        """
        Assess confidence in retrieval results
        
        Args:
            top_k_results: Top-k reranked results
        """
        if not top_k_results:
            return {
                "overall_confidence": 0.0,
                "score_range": 0.0,
                "top_result_score": 0.0,
                "score_variance": 0.0,
            }
        
        scores = [r.relevance_score for r in top_k_results]
        
        return {
            "overall_confidence": sum(scores) / len(scores),
            "score_range": max(scores) - min(scores),
            "top_result_score": scores[0],
            "score_variance": ReliabilityEstimator._calculate_variance(scores),
        }

    @staticmethod
    def describe_retrieval_scores(scores: List[float]) -> Dict[str, float]:
        """Describe ranking scores without misrepresenting similarity as probability."""
        if not scores:
            return {"mean_similarity": 0.0, "score_range": 0.0, "score_spread": 0.0}
        return {
            "mean_similarity": sum(scores) / len(scores),
            "score_range": max(scores) - min(scores),
            "score_spread": ReliabilityEstimator._calculate_variance(scores),
        }

    @staticmethod
    def answer_reliability(
        source_scores: List[float],
        citations: List[Citation],
        source_count: int,
        grounding_score: float,
        provider_generated: bool,
    ) -> AnswerReliability:
        """Return uncalibrated evidence indicators without claiming probability."""
        evidence_quality = sum(source_scores) / len(source_scores) if source_scores else 0.0
        citation_coverage = len({citation.chunk_id for citation in citations}) / source_count if source_count else 0.0
        if not provider_generated:
            return AnswerReliability(
                evidence_quality=evidence_quality,
                citation_coverage=citation_coverage,
                grounding_score=grounding_score,
                reliability_indicator=None,
                score_type="retrieval_only_evidence_indicators",
            )
        indicator = 0.0 if not citations else (evidence_quality + citation_coverage + grounding_score) / 3.0
        return AnswerReliability(
            evidence_quality=evidence_quality,
            citation_coverage=citation_coverage,
            grounding_score=grounding_score,
            reliability_indicator=indicator,
            score_type="uncalibrated_evidence_reliability_indicator",
        )
    
    @staticmethod
    def _calculate_variance(scores: List[float]) -> float:
        """Calculate variance of scores"""
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5  # Standard deviation
