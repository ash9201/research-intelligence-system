"""Compare baseline retrieval against deterministic query decomposition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.query.decomposer import QueryDecomposer
from src.retrieval import IndexManager
from src.reranking import CrossEncoderReranker
from src.models import RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "query_decomposition_challenge.json"
)

INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

INDEX_NAME = "attention_is_all_you_need"

TOP_K = 5
INITIAL_K = 10
RERANK_POOL_SIZE = 40


def load_challenge_queries() -> list[dict]:
    """Load all supported challenge-query categories."""
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    supported_categories = {
        "multi_part",
        "cross_section",
        "paraphrase_heavy",
        "synthesis_comparison",
        "unsupported",
    }

    return [
        query
        for query in benchmark["queries"]
        if query["category"] in supported_categories
    ]


def deduplicate_results(
    results: Iterable[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Deduplicate retrieval results by chunk ID.

    When the same chunk is retrieved for multiple subqueries, keep the
    highest-scoring occurrence.
    """
    best_by_chunk: dict[str, RetrievalResult] = {}

    for result in results:
        existing = best_by_chunk.get(result.chunk_id)

        if existing is None or result.score > existing.score:
            best_by_chunk[result.chunk_id] = result

    return sorted(
        best_by_chunk.values(),
        key=lambda result: result.score,
        reverse=True,
    )


def evaluate_top_k(
    results: list[RetrievalResult],
    relevant_chunk_ids: set[str],
    k: int,
    is_unsupported: bool = False,
) -> dict[str, float]:
    """Compute retrieval metrics for one query."""
    top_results = results[:k]
    retrieved_ids = [result.chunk_id for result in top_results]

    # Negative/unsupported queries are evaluated separately.
    if is_unsupported:
        false_positive_count = sum(
            1
            for chunk_id in retrieved_ids
            if chunk_id not in relevant_chunk_ids
        )

        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit": 0.0,
            "mrr": 0.0,
            "unsupported_false_positive_rate": (
                false_positive_count / len(top_results)
                if top_results
                else 0.0
            ),
        }

    relevant_retrieved = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_chunk_ids
    ]

    hit = 1.0 if relevant_retrieved else 0.0

    precision = (
        len(relevant_retrieved) / len(top_results)
        if top_results
        else 0.0
    )

    recall = (
        len(set(relevant_retrieved)) / len(relevant_chunk_ids)
        if relevant_chunk_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "hit": hit,
        "mrr": reciprocal_rank,
        "unsupported_false_positive_rate": 0.0,
    }

def evaluate_candidate_recall(
    results: list[RetrievalResult],
    relevant_chunk_ids: set[str],
) -> dict[str, float]:
    """Measure how much relevant evidence exists before reranking."""
    if not relevant_chunk_ids:
        return {
            "candidate_recall": 0.0,
            "candidate_relevant_count": 0.0,
            "candidate_count": float(len(results)),
        }

    retrieved_ids = {
        result.chunk_id
        for result in results
    }

    relevant_found = retrieved_ids.intersection(relevant_chunk_ids)

    return {
        "candidate_recall": (
            len(relevant_found) / len(relevant_chunk_ids)
        ),
        "candidate_relevant_count": float(len(relevant_found)),
        "candidate_count": float(len(results)),
    }

def rerank_results(
    reranker: CrossEncoderReranker,
    query: str,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Rerank merged candidate results using the original user query.

    This mirrors the architecture we intend to use in the real system:
    subqueries increase recall, but the original query determines final
    relevance.
    """
    if not results:
        return []

    reranked = reranker.rerank(
        query,
        results[:RERANK_POOL_SIZE],
        top_k=TOP_K,
    )

    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            content=result.content,
            score=result.relevance_score,
            retrieval_method="decomposition_reranked",
        )
        for result in reranked
    ]


def run_baseline(
    retriever,
    reranker: CrossEncoderReranker,
    query: str,
) -> list[RetrievalResult]:
    """Run the existing baseline retrieval pipeline."""
    initial = retriever.retrieve(
        query,
        top_k=INITIAL_K,
    )

    return rerank_results(
        reranker,
        query,
        initial,
    )


def run_decomposed(
    retriever,
    reranker: CrossEncoderReranker,
    query: str,
) -> tuple[
    list[RetrievalResult],
    list[RetrievalResult],
    list[str],
]:
    """
    Run retrieval independently for each subquery.

    Returns:
        candidate_results:
            Deduplicated candidates before reranking.
        final_results:
            Final top-k results after reranking with the original query.
        subquery_texts:
            Text of generated subqueries.
    """
    decomposition = QueryDecomposer.decompose(query)

    all_candidates: list[RetrievalResult] = []

    for subquery in decomposition.subqueries:
        results = retriever.retrieve(
            subquery.text,
            top_k=INITIAL_K,
        )
        all_candidates.extend(results)

    candidate_results = deduplicate_results(all_candidates)

    final_results = rerank_results(
        reranker,
        query,
        candidate_results,
    )

    subquery_texts = [
        subquery.text
        for subquery in decomposition.subqueries
    ]

    return candidate_results, final_results, subquery_texts


def print_results(
    query_id: str,
    query: str,
    relevant_ids: set[str],
    baseline_candidates: list[RetrievalResult],
    baseline_results: list[RetrievalResult],
    decomposed_candidates: list[RetrievalResult],
    decomposed_results: list[RetrievalResult],
    subqueries: list[str],
) -> None:
    """Print a readable per-query experiment result."""

    baseline_candidate_metrics = evaluate_candidate_recall(
        baseline_candidates,
        relevant_ids,
    )

    decomposed_candidate_metrics = evaluate_candidate_recall(
        decomposed_candidates,
        relevant_ids,
    )

    baseline_metrics = evaluate_top_k(
        baseline_results,
        relevant_ids,
        TOP_K,
    )

    decomposed_metrics = evaluate_top_k(
        decomposed_results,
        relevant_ids,
        TOP_K,
    )

    print("\n" + "=" * 90)
    print(f"{query_id}: {query}")
    print("-" * 90)

    print("Subqueries:")
    for subquery in subqueries:
        print(f"  - {subquery}")

    print("\nRelevant chunks:")
    for chunk_id in sorted(relevant_ids):
        print(f"  - {chunk_id}")

    print("\nBaseline top-k:")
    for rank, result in enumerate(baseline_results, start=1):
        marker = " <-- RELEVANT" if result.chunk_id in relevant_ids else ""
        print(
            f"  {rank}. {result.chunk_id}"
            f" | score={result.score:.4f}{marker}"
        )

    print("\nDecomposed top-k:")
    for rank, result in enumerate(decomposed_results, start=1):
        marker = " <-- RELEVANT" if result.chunk_id in relevant_ids else ""
        print(
            f"  {rank}. {result.chunk_id}"
            f" | score={result.score:.4f}{marker}"
        )
    
    print("\nCandidate-stage retrieval:")
    print(
        f"  Baseline:   "
        f"{int(baseline_candidate_metrics['candidate_relevant_count'])}/"
        f"{len(relevant_ids)} relevant chunks in "
        f"{int(baseline_candidate_metrics['candidate_count'])} candidates "
        f"(Recall={baseline_candidate_metrics['candidate_recall']:.3f})"
    )
    print(
        f"  Decomposed: "
        f"{int(decomposed_candidate_metrics['candidate_relevant_count'])}/"
        f"{len(relevant_ids)} relevant chunks in "
        f"{int(decomposed_candidate_metrics['candidate_count'])} candidates "
        f"(Recall={decomposed_candidate_metrics['candidate_recall']:.3f})"
    )
    print("\nMetrics:")
    print(
        f"  Baseline    "
        f"Precision@{TOP_K}={baseline_metrics['precision']:.3f}, "
        f"Recall@{TOP_K}={baseline_metrics['recall']:.3f}, "
        f"Hit={baseline_metrics['hit']:.0f}, "
        f"MRR={baseline_metrics['mrr']:.3f}"
    )
    print(
        f"  Decomposed  "
        f"Precision@{TOP_K}={decomposed_metrics['precision']:.3f}, "
        f"Recall@{TOP_K}={decomposed_metrics['recall']:.3f}, "
        f"Hit={decomposed_metrics['hit']:.0f}, "
        f"MRR={decomposed_metrics['mrr']:.3f}"
    )
    baseline_found = [
        result.chunk_id
        for result in baseline_results
        if result.chunk_id in relevant_ids
    ]

    decomposed_found = [
        result.chunk_id
        for result in decomposed_results
        if result.chunk_id in relevant_ids
    ]

    print("\nRelevant evidence retrieved:")
    print(f"  Baseline:   {len(baseline_found)}/{len(relevant_ids)}")
    for chunk_id in baseline_found:
        print(f"    - {chunk_id}")

    print(f"  Decomposed: {len(decomposed_found)}/{len(relevant_ids)}")
    for chunk_id in decomposed_found:
        print(f"    - {chunk_id}")


def main() -> None:
    """Run the decomposition experiment."""
    queries = load_challenge_queries()

    print(f"Loaded {len(queries)} challenge queries.")

    index_manager = IndexManager(INDEX_DIR)
    retriever = index_manager.load_index(INDEX_NAME)

    reranker = CrossEncoderReranker()

    aggregate_baseline = []
    aggregate_decomposed = []
    negative_results = []

    for query_data in queries:
        query_id = query_data["query_id"]
        query = query_data["query"]

        relevant_ids = set(query_data["relevant_chunk_ids"])

        baseline_candidates = retriever.retrieve(
            query,
            top_k=INITIAL_K,
        )

        baseline_results = rerank_results(
            reranker,
            query,
            baseline_candidates,
        )

        decomposed_candidates, decomposed_results, subqueries = run_decomposed(
            retriever,
            reranker,
            query,
        )
        
        is_unsupported = query_data["category"] == "unsupported"
        baseline_metrics = evaluate_top_k(
            baseline_results,
            relevant_ids,
            TOP_K,
            is_unsupported=is_unsupported,
        )

        decomposed_metrics = evaluate_top_k(
            decomposed_results,
            relevant_ids,
            TOP_K,
            is_unsupported=is_unsupported,
        )

        if is_unsupported:
            negative_results.append(
                {
                    "query_id": query_id,
                    "baseline_false_positive_rate": baseline_metrics[
                        "unsupported_false_positive_rate"
                    ],
                    "decomposed_false_positive_rate": decomposed_metrics[
                        "unsupported_false_positive_rate"
                    ],
                }
            )
        else:
            aggregate_baseline.append(baseline_metrics)
            aggregate_decomposed.append(decomposed_metrics)

        print_results(
            query_id,
            query,
            relevant_ids,
            baseline_candidates,
            baseline_results,
            decomposed_candidates,
            decomposed_results,
            subqueries,
        )

    if not queries:
        print("No multi-part queries found.")
        return

    def mean_metric(results: list[dict], key: str) -> float:
        return sum(item[key] for item in results) / len(results)

    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS — SUPPORTED QUERIES")
    print("=" * 90)

    for metric in ("precision", "recall", "hit", "mrr"):
        baseline_value = mean_metric(aggregate_baseline, metric)
        decomposed_value = mean_metric(aggregate_decomposed, metric)
        difference = decomposed_value - baseline_value

        print(
            f"{metric.upper():<10} "
            f"baseline={baseline_value:.4f}  "
            f"decomposed={decomposed_value:.4f}  "
            f"delta={difference:+.4f}"
        )
    if negative_results:
        print("\n" + "=" * 90)
        print("UNSUPPORTED / NEGATIVE QUERY RESULTS")
        print("=" * 90)

        for result in negative_results:
            print(
                f"{result['query_id']}: "
                f"baseline false-positive rate="
                f"{result['baseline_false_positive_rate']:.3f}, "
                f"decomposed false-positive rate="
                f"{result['decomposed_false_positive_rate']:.3f}"
            )


if __name__ == "__main__":
    main()