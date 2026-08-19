"""
Grounding and answer evaluation
"""
import re
from typing import Dict, List, Tuple

from src.logging_config import get_logger
from src.models import Citation

logger = get_logger(__name__)


class GroundingEvaluator:
    """Evaluates grounding quality of answers"""
    
    @staticmethod
    def evaluate_citation_coverage(
        citations: List[Citation],
        answer_tokens: List[str],
    ) -> float:
        """
        Evaluate how well citations cover the answer
        
        Args:
            citations: Extracted citations
            answer_tokens: Tokens in the answer
        """
        if not citations or not answer_tokens:
            return 0.0
        
        # Count tokens covered by citations
        covered_tokens = 0
        for citation in citations:
            for start, end in citation.position_in_answer:
                # This is a simplified metric
                covered_tokens += 1
        
        return min(covered_tokens / len(answer_tokens), 1.0)
    
    @staticmethod
    def evaluate_citation_accuracy(
        citations: List[Citation],
        source_documents: dict,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate if citations match source content
        
        Args:
            citations: Extracted citations
            source_documents: Map of doc_id to document content
        """
        errors = []
        accurate_citations = 0
        
        for citation in citations:
            if citation.doc_id in source_documents:
                source = source_documents[citation.doc_id]
                # Check if cited content appears in source
                if citation.content in source or citation.content[:50] in source:
                    accurate_citations += 1
                else:
                    errors.append(f"Citation mismatch: {citation.content[:50]}...")
            else:
                errors.append(f"Unknown document: {citation.doc_id}")
        
        accuracy = accurate_citations / len(citations) if citations else 0.0
        return accuracy, errors
    
    @staticmethod
    def assess_hallucination(
        answer: str,
        sources: List[Citation],
        threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """
        Assess if answer might contain hallucinations
        
        Args:
            answer: Generated answer
            sources: Supporting citations
            threshold: Hallucination threshold (if coverage < threshold)
        """
        if not sources:
            # No sources means potential hallucination
            return True, 0.0
        
        answer_tokens = set(answer.lower().split())
        source_tokens = set()
        
        for source in sources:
            source_tokens.update(source.content.lower().split())
        
        # Calculate overlap
        overlap = len(answer_tokens & source_tokens) / len(answer_tokens) if answer_tokens else 0
        
        is_hallucinating = overlap < threshold
        return is_hallucinating, overlap

    @staticmethod
    def evaluate_claim_support(answer: str, sources: List[Citation]) -> Dict[str, float]:
        """Measure sentence-level lexical support; this is not probability calibration."""
        claims = [claim.strip() for claim in re.split(r"(?<=[.!?])\s+", answer) if claim.strip()]
        source_tokens = {
            token
            for source in sources
            for token in re.findall(r"\w+", source.content.lower())
        }
        supported = 0
        for claim in claims:
            claim_tokens = set(re.findall(r"\w+", claim.lower()))
            if claim_tokens and len(claim_tokens & source_tokens) / len(claim_tokens) >= 0.5:
                supported += 1
        return {
            "claim_count": float(len(claims)),
            "supported_claim_count": float(supported),
            "support_ratio": supported / len(claims) if claims else 0.0,
        }

    @staticmethod
    def evaluate_retrieval_source_agreement(
        retrieved_chunk_ids: List[str],
        citations: List[Citation],
    ) -> float:
        """Return the fraction of citations that refer to retrieved evidence."""
        if not citations:
            return 0.0
        retrieved = set(retrieved_chunk_ids)
        return sum(citation.chunk_id in retrieved for citation in citations) / len(citations)

    @staticmethod
    def validate_citation_support(
        answer: str,
        citations: List[Citation],
        minimum_overlap: float = 0.2,
    ) -> Tuple[List[Citation], List[Citation]]:
        """Split citations into lexically supported and unsupported claim references.

        This is a conservative lexical gate, not semantic entailment or calibration.
        It prevents a source marker appended to unrelated text from counting as
        grounded while leaving stronger evaluators replaceable later.
        """
        supported = []
        unsupported = []
        for citation in citations:
            source_tokens = GroundingEvaluator._meaningful_tokens(citation.content)
            claim_supported = False
            for start, end in citation.position_in_answer:
                claim_text = GroundingEvaluator._claim_near_marker(answer, start, end)
                claim_tokens = GroundingEvaluator._meaningful_tokens(claim_text)
                if claim_tokens and len(claim_tokens & source_tokens) / len(claim_tokens) >= minimum_overlap:
                    claim_supported = True
                    break
            (supported if claim_supported else unsupported).append(citation)
        return supported, unsupported

    @staticmethod
    def _claim_near_marker(answer: str, start: int, end: int) -> str:
        """Extract the claim before or after a marker, including punctuation-before-marker."""
        before = answer[:start].rstrip()
        if before and before[-1] in ".!?":
            before = before[:-1].rstrip()
        previous_boundary = max(
            before.rfind("."),
            before.rfind("!"),
            before.rfind("?"),
        )
        left = before[previous_boundary + 1:]

        after = answer[end:]
        next_boundaries = [
            position for position in (
                after.find("."),
                after.find("!"),
                after.find("?"),
            ) if position >= 0
        ]
        right = after[:min(next_boundaries)] if next_boundaries else after
        return f"{left} {right}"

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        """Return content-bearing tokens for the conservative support check."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
            "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
            "with", "source", "answer",
        }
        return {
            token for token in re.findall(r"[a-z0-9_]+", text.lower())
            if token not in stopwords and len(token) > 2
        }
