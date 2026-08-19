"""Run the manually labeled Attention Is All You Need retrieval benchmark."""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.chunking import DocumentChunker
from src.evaluation.metrics import EvaluationFramework
from src.ingestion import DocumentLoader
from src.metadata import MetadataExtractor
from src.reranking import CrossEncoderReranker
from src.retrieval import IndexManager


METRIC_KEYS = ["precision@1", "precision@5", "recall@5", "recall@10", "hit_rate@5", "mrr", "ndcg@5", "ndcg@10"]


def mean_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Average measured metrics over an explicitly supplied group of query rows."""
    rows = list(rows)
    return {
        key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
        for key in [*METRIC_KEYS, "latency_ms"]
    }


def best_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> int | None:
    """Return the first relevant rank, or None when no relevant label is retrieved."""
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return index
    return None


def classify_reranker_impact(
    relevant_ids: List[str],
    before_ids: List[str],
    after_ids: List[str],
) -> Dict[str, Any]:
    """Classify reranking by its measured best relevant rank; negatives are excluded."""
    if not relevant_ids:
        return {"classification": "not_applicable_negative", "before_rank": None, "after_rank": None}
    before_rank = best_rank(relevant_ids, before_ids)
    after_rank = best_rank(relevant_ids, after_ids)
    if before_rank is not None and after_rank is None:
        classification = "left_top_k"
    elif before_rank is None and after_rank is not None:
        classification = "improved_into_top_k"
    elif before_rank is None:
        classification = "unchanged_outside_top_k"
    elif after_rank < before_rank:
        classification = "improved"
    elif after_rank > before_rank:
        classification = "worse"
    else:
        classification = "unchanged"
    return {"classification": classification, "before_rank": before_rank, "after_rank": after_rank}


def retrieve_variant(retriever, reranker, query: str, variant: str) -> tuple[List[str], List[str]]:
    """Return final IDs plus weighted-hybrid candidate IDs for the fixed variant set."""
    if variant == "bm25":
        results = retriever.bm25.retrieve(query, top_k=20)
        return [result.chunk_id for result in results], []
    if variant == "dense":
        results = retriever.dense.retrieve(query, top_k=20)
        return [result.chunk_id for result in results], []

    retriever.fusion_strategy = "rrf" if variant == "rrf_hybrid" else "weighted"
    hybrid_results = retriever.retrieve(query, top_k=20, bm25_k=20, dense_k=20)
    hybrid_ids = [result.chunk_id for result in hybrid_results]
    if variant != "hybrid_reranker":
        return hybrid_ids, hybrid_ids
    reranked = reranker.rerank(query, hybrid_results, top_k=10)
    return [result.chunk_id for result in reranked], hybrid_ids


def run_variant(retriever, reranker, queries: List[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    """Measure one unmodified retrieval configuration on the frozen labels."""
    rows = []
    for benchmark_query in queries:
        started = time.perf_counter()
        retrieved_ids, candidate_ids = retrieve_variant(retriever, reranker, benchmark_query["query"], variant)
        latency_ms = (time.perf_counter() - started) * 1000
        relevant_ids = benchmark_query["relevant_chunk_ids"]
        metrics = EvaluationFramework.evaluate_retrieval(relevant_ids, retrieved_ids, [1, 5, 10])
        rows.append(
            {
                "variant": variant,
                "query_id": benchmark_query["query_id"],
                "category": benchmark_query["category"],
                "query": benchmark_query["query"],
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "candidate_chunk_ids": candidate_ids,
                "latency_ms": latency_ms,
                **{key: metrics[key] for key in METRIC_KEYS},
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows, serializing list fields as JSON to preserve machine readability."""
    normalized_rows = [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized_rows[0]))
        writer.writeheader()
        writer.writerows(normalized_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed retrieval variants on Attention Is All You Need.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmarks/attention_is_all_you_need_retrieval.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments"))
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    document = DocumentLoader.load_document(Path(benchmark["document"]))
    chunking = benchmark["chunking"]
    chunks = DocumentChunker(**{"strategy": chunking["strategy"], "chunk_size": chunking["chunk_size"], "overlap": chunking["overlap"]}).chunk_pages(
        document.metadata["pages"], document.doc_id, {"title": document.title}
    )
    for chunk in chunks:
        MetadataExtractor.enrich_chunk_metadata(chunk)

    run_dir = args.output / f"{benchmark['name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    retriever = IndexManager(run_dir / "indexes").create_index("attention", chunks)
    reranker = CrossEncoderReranker()
    variants = ["bm25", "dense", "weighted_hybrid", "rrf_hybrid", "hybrid_reranker"]
    variant_rows = {variant: run_variant(retriever, reranker, benchmark["queries"], variant) for variant in variants}

    overall = [
        {"variant": variant, **mean_metrics(rows)}
        for variant, rows in variant_rows.items()
    ]
    category_rows = []
    for variant, rows in variant_rows.items():
        by_category = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        category_rows.extend(
            {"variant": variant, "category": category, "query_count": len(group), **mean_metrics(group)}
            for category, group in sorted(by_category.items())
        )

    weighted_rows = {row["query_id"]: row for row in variant_rows["weighted_hybrid"]}
    reranked_rows = {row["query_id"]: row for row in variant_rows["hybrid_reranker"]}
    reranker_impact = []
    for query in benchmark["queries"]:
        before = weighted_rows[query["query_id"]]["retrieved_chunk_ids"]
        after = reranked_rows[query["query_id"]]["retrieved_chunk_ids"]
        reranker_impact.append(
            {
                "query_id": query["query_id"],
                "category": query["category"],
                "query": query["query"],
                "relevant_chunk_ids": query["relevant_chunk_ids"],
                "weighted_hybrid_ids": before,
                "reranked_ids": after,
                **classify_reranker_impact(query["relevant_chunk_ids"], before[:10], after[:10]),
            }
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": {key: value for key, value in benchmark.items() if key != "queries"},
        "query_count": len(benchmark["queries"]),
        "variants": variant_rows,
        "overall_comparison": overall,
        "category_comparison": category_rows,
        "reranker_impact": reranker_impact,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(run_dir / "per_query_results.csv", [row for rows in variant_rows.values() for row in rows])
    write_csv(run_dir / "overall_comparison.csv", overall)
    write_csv(run_dir / "category_comparison.csv", category_rows)
    write_csv(run_dir / "reranker_impact.csv", reranker_impact)

    impact_counts = defaultdict(int)
    for row in reranker_impact:
        impact_counts[row["classification"]] += 1
    report = [
        "# Attention Is All You Need Retrieval Benchmark",
        "",
        benchmark["description"],
        "",
        "## Overall Comparison",
        "",
        "| Variant | P@1 | P@5 | R@5 | R@10 | Hit@5 | MRR | NDCG@5 | NDCG@10 | Mean latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['variant']} | {row['precision@1']:.4f} | {row['precision@5']:.4f} | {row['recall@5']:.4f} | {row['recall@10']:.4f} | {row['hit_rate@5']:.4f} | {row['mrr']:.4f} | {row['ndcg@5']:.4f} | {row['ndcg@10']:.4f} | {row['latency_ms']:.2f} |"
        for row in overall
    )
    report.extend(["", "## Query Category Comparison", "", "| Variant | Category | Queries | P@1 | R@5 | Hit@5 | MRR | NDCG@10 | Mean latency ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    report.extend(
        f"| {row['variant']} | {row['category']} | {row['query_count']} | {row['precision@1']:.4f} | {row['recall@5']:.4f} | {row['hit_rate@5']:.4f} | {row['mrr']:.4f} | {row['ndcg@10']:.4f} | {row['latency_ms']:.2f} |"
        for row in category_rows
    )
    report.extend(["", "## Latency Comparison", "", "| Variant | Mean retrieval latency ms |", "|---|---:|"])
    report.extend(f"| {row['variant']} | {row['latency_ms']:.2f} |" for row in overall)
    report.extend(["", "## Reranker Impact", ""])
    report.extend(f"- {name}: {count}" for name, count in sorted(impact_counts.items()))
    report.extend(["", "## Artifacts", "", "- `results.json`: complete machine-readable result set", "- `per_query_results.csv`: per-query metrics for all variants", "- `overall_comparison.csv`: aggregate comparison", "- `category_comparison.csv`: category breakdown", "- `reranker_impact.csv`: weighted-hybrid vs reranker rank deltas"])
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()