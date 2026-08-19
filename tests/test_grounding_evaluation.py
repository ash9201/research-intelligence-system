"""Tests for modular answer-evidence reliability signals."""
from src.evaluation import GroundingEvaluator
from src.models import Citation


def test_claim_support_and_retrieval_source_agreement():
    """Support and agreement report evidence alignment independently of retrieval score."""
    citation = Citation(
        chunk_id="chunk_1",
        doc_id="doc_1",
        content="Machine learning learns patterns from training data.",
        relevance_score=0.8,
        position_in_answer=[(0, 10)],
    )
    support = GroundingEvaluator.evaluate_claim_support(
        "Machine learning learns patterns from training data. It predicts weather.", [citation]
    )

    assert support["claim_count"] == 2
    assert support["supported_claim_count"] == 1
    assert GroundingEvaluator.evaluate_retrieval_source_agreement(["chunk_1"], [citation]) == 1.0