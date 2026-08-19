"""
Grounding and citation extraction
"""
import re
from typing import List, Tuple

from src.logging_config import get_logger
from src.models import Citation, EvidenceSource

logger = get_logger(__name__)


class GroundingExtractor:
    """Extracts citations and groundings from generated answers"""
    
    @staticmethod
    def extract_citations(
        answer: str,
        sources: List[EvidenceSource],
    ) -> List[Citation]:
        """
        Extract citations from answer
        
        Args:
            answer: Generated answer text
            sources: Source documents
        """
        citations = []
        
        # Look for [Source N] references in the answer
        source_refs = re.findall(r'\[Source (\d+)\]', answer)
        
        for source_ref in dict.fromkeys(source_refs):
            try:
                source_idx = int(source_ref) - 1
                if 0 <= source_idx < len(sources):
                    source = sources[source_idx]
                    
                    # Find positions of this source reference in answer
                    pattern = f"\\[Source {source_ref}\\]"
                    positions = [
                        (m.start(), m.end())
                        for m in re.finditer(pattern, answer)
                    ]
                    
                    citation = Citation(
                        chunk_id=source.chunk_id,
                        doc_id=source.doc_id,
                        content=source.content,
                        relevance_score=source.score,
                        position_in_answer=positions,
                    )
                    citations.append(citation)
            except (ValueError, IndexError):
                logger.warning(f"Invalid source reference: [Source {source_ref}]")
        
        logger.info(f"Extracted {len(citations)} citations from answer")
        return citations
    
    @staticmethod
    def clean_answer(answer: str) -> str:
        """Preserve Markdown and strict source markers for rendered traceability."""
        return answer.strip()
    
    @staticmethod
    def assess_groundedness(
        answer: str,
        citations: List[Citation],
        min_citations: int = 1,
    ) -> Tuple[float, str]:
        """
        Assess how well-grounded an answer is
        
        Args:
            answer: Generated answer
            citations: Extracted citations
            min_citations: Minimum expected citations
        """
        if len(citations) < min_citations:
            confidence = 0.5
            reason = f"Insufficient citations (found {len(citations)}, expected {min_citations})"
        elif len(citations) >= 3:
            confidence = 0.9
            reason = "Well-grounded with multiple sources"
        else:
            confidence = 0.7
            reason = "Grounded but could benefit from more sources"
        
        # Check if answer has significant content
        if len(answer.split()) < 20:
            confidence *= 0.7
            reason = f"{reason} (brief answer)"
        
        return confidence, reason
