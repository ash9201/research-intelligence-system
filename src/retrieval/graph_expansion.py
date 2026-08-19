"""Graph-expanded candidate retrieval layered on an existing hybrid retriever."""
from typing import Dict, List, Optional

from src.graph import KnowledgeGraph
from src.models import Chunk, RetrievalResult
from src.retrieval.hybrid import HybridRetriever


class GraphExpandedRetriever:
    """Adds chunks from citation-neighbor documents to base retrieval candidates."""

    def __init__(
        self,
        base_retriever: HybridRetriever,
        graph: KnowledgeGraph,
        chunks: Optional[List[Chunk]] = None,
        direction: str = "both",
        depth: int = 1,
        expansion_decay: float = 0.5,
    ):
        if not 0 < expansion_decay <= 1:
            raise ValueError("expansion_decay must be in (0, 1]")
        self.base_retriever = base_retriever
        self.graph = graph
        self.direction = direction
        self.depth = depth
        self.expansion_decay = expansion_decay
        source_chunks = chunks or list(base_retriever.bm25.chunk_map.values())
        self.chunks_by_doc: Dict[str, List[Chunk]] = {}
        for chunk in source_chunks:
            self.chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
        expand: bool = True,
        direction: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve base candidates and optionally add citation-neighbor document chunks."""
        base_results = self.base_retriever.retrieve(query, top_k=candidate_k)
        if not expand or not base_results:
            return base_results[:top_k]

        seed_scores = {}
        for result in base_results:
            seed_scores[result.doc_id] = max(seed_scores.get(result.doc_id, 0.0), result.score)
        active_direction = direction or self.direction
        active_depth = depth or self.depth
        expanded_docs = self.graph.expand_documents(
            list(seed_scores), direction=active_direction, depth=active_depth
        )
        candidates = {result.chunk_id: result for result in base_results}
        for document_id in expanded_docs:
            inherited_score = max(seed_scores.values(), default=0.0) * self.expansion_decay
            for chunk in self.chunks_by_doc.get(document_id, []):
                candidates.setdefault(
                    chunk.chunk_id,
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        score=inherited_score,
                        retrieval_method="graph_expanded",
                        metadata={
                            **chunk.metadata,
                            "graph_direction": active_direction,
                            "graph_depth": active_depth,
                            "graph_expansion_score": inherited_score,
                        },
                    ),
                )
        return sorted(candidates.values(), key=lambda result: result.score, reverse=True)[:top_k]