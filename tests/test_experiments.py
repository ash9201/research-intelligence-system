"""Tests for reproducible experiments and benchmark result artifacts."""
from pathlib import Path

from src.evaluation import RetrievalBenchmark
from src.experiments import RetrievalExperimentConfig
from src.models import RetrievalResult


class StubRetriever:
    """Deterministic retriever used to test evaluation mechanics, not quality."""

    def __init__(self):
        self.alpha = 0.5
        self.fusion_strategy = "weighted"
        self.rrf_k = 60
        self.bm25 = self
        self.dense = self

    def retrieve(self, query, top_k=10, **kwargs):
        return [
            RetrievalResult(chunk_id="chunk_1", doc_id="doc_1", content="relevant", score=0.9, retrieval_method="stub"),
            RetrievalResult(chunk_id="chunk_2", doc_id="doc_2", content="other", score=0.2, retrieval_method="stub"),
        ][:top_k]


def test_experiment_config_round_trips(tmp_path):
    """A saved configuration preserves every retrieval-affecting setting."""
    config = RetrievalExperimentConfig(
        name="weighted_hybrid",
        chunking_method="sentence",
        fusion_strategy="rrf",
        graph_expansion_enabled=True,
        graph_direction="upstream",
    )
    path = tmp_path / "experiment.json"
    config.save(path)

    assert RetrievalExperimentConfig.load(path) == config


def test_benchmark_writes_json_csv_and_report(tmp_path):
    """Benchmark metrics are derived from labeled results and emitted as artifacts."""
    benchmark = RetrievalBenchmark(
        {
            "name": "unit_benchmark",
            "description": "Deterministic evaluation fixture.",
            "queries": [{"query_id": "q1", "query": "test", "relevant_doc_ids": ["doc_1"]}],
        }
    )
    result = benchmark.run(StubRetriever(), RetrievalExperimentConfig(name="bm25"), k_values=[1])
    paths = benchmark.write_results(result, tmp_path / "output")

    assert result["aggregate"]["precision@1"] == 1.0
    assert result["aggregate"]["mean_latency_ms"] >= 0
    assert all(path.exists() for path in paths.values())