"""Evidence normalization and retrieval-only summaries for the Ask pipeline."""
import re
from typing import Iterable, List

from src.models import EvidenceSource


def normalize_evidence(selected_results: Iterable, chunk_metadata: dict) -> List[EvidenceSource]:
    """Normalize retrieval or reranking results into one generation-source contract."""
    evidence = []
    for index, result in enumerate(selected_results, start=1):
        metadata = chunk_metadata.get(result.chunk_id, {})
        evidence.append(EvidenceSource(
            source_index=index,
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            content=result.content,
            score=float(getattr(result, "relevance_score", getattr(result, "score", 0.0))),
            title=metadata.get("title"),
            page=metadata.get("page"),
            pages=metadata.get("pages", []),
            section=metadata.get("section"),
        ))
    return evidence


def retrieval_only_summary(sources: List[EvidenceSource], max_sources: int = 3) -> str:
    """Return an explicitly labeled, sentence-bounded evidence summary with markers."""
    parts = ["## Retrieval-only evidence summary", "", "No provider-generated synthesis is available. Relevant evidence:"]
    for source in sources[:max_sources]:
        parts.append(f"- {_first_complete_sentence(source.content)} [Source {source.source_index}]")
    return "\n".join(parts)


def _first_complete_sentence(content: str) -> str:
    """Choose a complete first sentence without arbitrary character slicing."""
    normalized = re.sub(r"\s+", " ", content).strip()
    match = re.search(r".+?[.!?](?:\s|$)", normalized)
    return (match.group(0).strip() if match else normalized).strip()