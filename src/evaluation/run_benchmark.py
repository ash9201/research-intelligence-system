"""CLI for executing the labeled local demonstration benchmark."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import DocumentChunker
from src.evaluation import RetrievalBenchmark
from src.experiments import RetrievalExperimentConfig
from src.models import Chunk
from src.retrieval import IndexManager


def build_chunks(benchmark_data: dict, config: RetrievalExperimentConfig) -> list[Chunk]:
    """Build deterministic chunks from the benchmark's versioned document corpus."""
    chunker = DocumentChunker(
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
        strategy=config.chunking_method,
    )
    chunks = []
    for document in benchmark_data.get("documents", []):
        chunks.extend(
            chunker.chunk_document(
                document["content"],
                document["doc_id"],
                metadata={"title": document.get("title", document["doc_id"]), "benchmark": True},
            )
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the labeled local retrieval benchmark.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmarks/project_benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments"))
    args = parser.parse_args()

    benchmark_data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    configurations = [
        RetrievalExperimentConfig(name="bm25"),
        RetrievalExperimentConfig(name="dense"),
        RetrievalExperimentConfig(name="weighted_hybrid", fusion_strategy="weighted"),
        RetrievalExperimentConfig(name="rrf_hybrid", fusion_strategy="rrf"),
        RetrievalExperimentConfig(name="hybrid_reranker", fusion_strategy="weighted", reranker_enabled=True),
    ]
    run_dir = args.output / f"{benchmark_data['name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    benchmark = RetrievalBenchmark(benchmark_data)

    for config in configurations:
        chunks = build_chunks(benchmark_data, config)
        index_manager = IndexManager(run_dir / "indexes")
        retriever = index_manager.create_index(
            config.name,
            chunks,
            embedding_model=config.embedding_model,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
            fusion_strategy=config.fusion_strategy,
            hybrid_alpha=config.hybrid_alpha,
            rrf_k=config.rrf_k,
        )
        result = benchmark.run(retriever, config)
        output_paths = benchmark.write_results(result, run_dir / config.name)
        config.save(run_dir / config.name / "experiment.json")
        print(f"{config.name}: {output_paths['report']}")


if __name__ == "__main__":
    main()