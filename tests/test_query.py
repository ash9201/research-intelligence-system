"""Tests for query analysis and decomposition."""

from src.query.analyzer import QueryAnalyzer
from src.query.decomposer import QueryDecomposer


def test_simple_query_is_not_decomposed():
    query = "What is multi-head attention?"

    result = QueryAnalyzer.analyze(query)

    assert result.complexity == "simple"
    assert result.requires_decomposition is False


def test_multi_part_query_is_detected():
    query = (
        "What optimizer, learning-rate schedule, and warmup strategy "
        "were used to train the Transformer?"
    )

    result = QueryAnalyzer.analyze(query)

    assert result.complexity == "multi_part"
    assert result.requires_decomposition is True


def test_empty_query_is_invalid():
    result = QueryAnalyzer.analyze("   ")

    assert result.complexity == "invalid"
    assert result.requires_decomposition is False


def test_optimizer_query_decomposes():
    query = (
        "What optimizer, learning-rate schedule, and warmup strategy "
        "were used to train the Transformer?"
    )

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 3
    assert "optimizer" in result.subqueries[0].text.lower()
    assert "learning-rate schedule" in result.subqueries[1].text.lower()
    assert "warmup strategy" in result.subqueries[2].text.lower()


def test_comparison_query_decomposes():
    query = "How does multi-head attention differ from single-head attention, and why?"

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 2
    assert "differ" in result.subqueries[0].text.lower()
    assert result.subqueries[1].text.lower().startswith("why")


def test_fallback_preserves_original_query():
    query = "What is scaled dot-product attention?"

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 1
    assert result.subqueries[0].text == query
    assert result.subqueries[0].query_id == "q1"

def test_three_part_question_is_fully_decomposed():
    query = (
        "What specific activation function is used in the "
        "position-wise feed-forward networks, how is the output "
        "of the multi-head attention concatenated, and why are "
        "the dot products scaled by 1/sqrt(d_k)?"
    )

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 3
    assert "activation function" in result.subqueries[0].text.lower()
    assert "concatenated" in result.subqueries[1].text.lower()
    assert "scaled" in result.subqueries[2].text.lower()


def test_two_part_question_is_decomposed():
    query = (
        "How does the network prevent information from flowing "
        "backwards in the auto-regressive generation phase, and "
        "what prevents over-fitting in the sums of the embeddings?"
    )

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 2
    assert "prevent information" in result.subqueries[0].text.lower()
    assert "over-fitting" in result.subqueries[1].text.lower()


def test_multi_part_optimizer_question_is_fully_decomposed():
    query = (
        "What regularization dropout rate was applied to the base model, "
        "how was the learning rate modified over the course of training, "
        "and why was label smoothing incorporated despite its negative "
        "effect on perplexity?"
    )

    result = QueryDecomposer.decompose(query)

    assert len(result.subqueries) == 3
    assert "dropout" in result.subqueries[0].text.lower()
    assert "learning rate" in result.subqueries[1].text.lower()
    assert "label smoothing" in result.subqueries[2].text.lower()