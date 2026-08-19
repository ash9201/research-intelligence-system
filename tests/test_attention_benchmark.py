"""Tests for the manually labeled Attention benchmark definition and analysis helpers."""
import json
from pathlib import Path

from src.evaluation.run_attention_benchmark import classify_reranker_impact, mean_metrics


def test_attention_benchmark_has_manual_category_coverage():
    """The paper benchmark is a fixed label set, not retrieval-derived labels."""
    benchmark = json.loads(Path("data/benchmarks/attention_is_all_you_need_retrieval.json").read_text())
    categories = {query["category"] for query in benchmark["queries"]}

    assert 25 <= len(benchmark["queries"]) <= 30
    assert categories == {
        "exact_lexical", "semantic_paraphrase", "mathematical_technical", "definition",
        "comparison", "multi_part", "cross_section", "broad", "unsupported_negative",
    }
    assert all("relevant_chunk_ids" in query for query in benchmark["queries"])
    assert sum(not query["relevant_chunk_ids"] for query in benchmark["queries"]) == 2


def test_reranker_impact_classification_uses_explicit_labels():
    """Rank-delta labels distinguish improvement, regression, and top-k loss."""
    assert classify_reranker_impact(["a"], ["x", "a"], ["a"])["classification"] == "improved"
    assert classify_reranker_impact(["a"], ["a"], ["x"])["classification"] == "left_top_k"
    assert classify_reranker_impact([], ["a"], ["a"])["classification"] == "not_applicable_negative"
    metrics = mean_metrics([{"precision@1": 1.0, "precision@5": 0.2, "recall@5": 1.0, "recall@10": 1.0, "hit_rate@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0, "latency_ms": 10.0}])
    assert metrics["latency_ms"] == 10.0