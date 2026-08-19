"""Reproducible benchmark execution over configured retrieval variants."""
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.evaluation.metrics import EvaluationFramework
from src.experiments import RetrievalExperimentConfig
from src.models import RetrievalResult
from src.reranking import CrossEncoderReranker
from src.retrieval import HybridRetriever


class RetrievalBenchmark:
    """Evaluate BM25, dense, fusion, and reranked variants on labeled queries."""

    def __init__(self, benchmark: Dict[str, Any]):
        self.benchmark = benchmark
        self.queries = benchmark.get("queries", [])
        if not self.queries:
            raise ValueError("benchmark must contain at least one query")

    @classmethod
    def load(cls, path: Path) -> "RetrievalBenchmark":
        """Load a project benchmark JSON file with query relevance labels."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def run(
        self,
        retriever: HybridRetriever,
        config: RetrievalExperimentConfig,
        k_values: Iterable[int] = (1, 3, 5),
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> Dict[str, Any]:
        """Run one configured variant and return per-query plus aggregate metrics."""
        per_query = []
        for benchmark_query in self.queries:
            query = benchmark_query["query"]
            relevant_chunk_ids = benchmark_query.get("relevant_chunk_ids")
            relevant_ids = relevant_chunk_ids or benchmark_query.get("relevant_doc_ids", [])
            started = time.perf_counter()
            results = self._retrieve_variant(retriever, query, config, reranker)
            latency_ms = (time.perf_counter() - started) * 1000
            retrieved_ids = [result.chunk_id for result in results] if relevant_chunk_ids else [result.doc_id for result in results]
            metrics = EvaluationFramework.evaluate_retrieval(list(relevant_ids), retrieved_ids, list(k_values))
            per_query.append(
                {
                    "query_id": benchmark_query.get("query_id", query),
                    "query": query,
                    "relevant_ids": list(relevant_ids),
                    "retrieved_ids": retrieved_ids,
                    "latency_ms": latency_ms,
                    **metrics,
                }
            )

        metric_keys = [key for key in per_query[0] if key in {"mrr"} or "@" in key]
        aggregate = {
            key: sum(item[key] for item in per_query) / len(per_query)
            for key in metric_keys
        }
        aggregate["mean_latency_ms"] = sum(item["latency_ms"] for item in per_query) / len(per_query)
        return {
            "benchmark_name": self.benchmark.get("name", "unnamed_benchmark"),
            "benchmark_description": self.benchmark.get("description", ""),
            "experiment": config.model_dump(mode="json"),
            "aggregate": aggregate,
            "per_query": per_query,
        }

    def run_configurations(
        self,
        retriever: HybridRetriever,
        configurations: Iterable[RetrievalExperimentConfig],
        k_values: Iterable[int] = (1, 3, 5),
    ) -> List[Dict[str, Any]]:
        """Run named configurations against identical labels and query order."""
        return [self.run(retriever, config, k_values=k_values) for config in configurations]

    @staticmethod
    def write_results(result: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Write machine-readable JSON/CSV and a concise Markdown report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "results.json"
        csv_path = output_dir / "per_query.csv"
        report_path = output_dir / "report.md"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        rows = result["per_query"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        lines = [
            f"# Retrieval Evaluation: {result['benchmark_name']}",
            "",
            result["benchmark_description"],
            "",
            "## Aggregate Metrics",
            "",
        ]
        lines.extend(f"- {name}: {value:.4f}" for name, value in result["aggregate"].items())
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"json": json_path, "csv": csv_path, "report": report_path}

    @staticmethod
    def _retrieve_variant(
        retriever: HybridRetriever,
        query: str,
        config: RetrievalExperimentConfig,
        reranker: Optional[CrossEncoderReranker],
    ) -> List[RetrievalResult]:
        candidate_k = max(config.bm25_top_k, config.dense_top_k, config.reranker_top_k)
        base_retriever = getattr(retriever, "base_retriever", retriever)
        if config.name == "bm25":
            return base_retriever.bm25.retrieve(query, top_k=config.bm25_top_k)
        if config.name == "dense":
            return base_retriever.dense.retrieve(query, top_k=config.dense_top_k)

        base_retriever.alpha = config.hybrid_alpha
        base_retriever.fusion_strategy = config.fusion_strategy
        base_retriever.rrf_k = config.rrf_k
        if config.graph_expansion_enabled and hasattr(retriever, "base_retriever"):
            results = retriever.retrieve(
                query,
                top_k=candidate_k,
                candidate_k=candidate_k,
                expand=True,
                direction=config.graph_direction,
                depth=config.graph_depth,
            )
        else:
            results = base_retriever.retrieve(
                query,
                top_k=candidate_k,
                bm25_k=config.bm25_top_k,
                dense_k=config.dense_top_k,
            )
        if config.reranker_enabled:
            if reranker is None:
                reranker = CrossEncoderReranker(model_name=config.reranker_model)
            reranked = reranker.rerank(query, results, top_k=config.reranker_top_k)
            return [
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    content=result.content,
                    score=result.relevance_score,
                    retrieval_method="hybrid_reranked",
                )
                for result in reranked
            ]
        return results