"""Tests for graph-aware candidate expansion."""
from src.graph import KnowledgeGraph
from src.models import Chunk, RetrievalResult
from src.retrieval import GraphExpandedRetriever


class StubHybridRetriever:
    """Minimal base retriever that exposes indexed chunks for graph expansion."""

    def __init__(self, chunks):
        self.bm25 = type("BM25", (), {"chunk_map": dict(enumerate(chunks))})()

    def retrieve(self, query, top_k=10):
        return [
            RetrievalResult(
                chunk_id="a_0",
                doc_id="doc_a",
                content="seed evidence",
                score=0.9,
                retrieval_method="hybrid_weighted",
            )
        ][:top_k]


def test_graph_expansion_adds_downstream_document_chunks():
    """Downstream citations become graph-expanded candidates after base retrieval."""
    chunks = [
        Chunk(chunk_id="a_0", doc_id="doc_a", content="seed evidence", start_char=0, end_char=13, chunk_index=0),
        Chunk(chunk_id="b_0", doc_id="doc_b", content="cited evidence", start_char=0, end_char=14, chunk_index=0),
    ]
    graph = KnowledgeGraph()
    graph.add_citation_relation("doc_a", "doc_b", "supports claim")

    retriever = GraphExpandedRetriever(
        StubHybridRetriever(chunks), graph, direction="downstream", depth=1
    )
    results = retriever.retrieve("seed", top_k=2)

    assert [result.doc_id for result in results] == ["doc_a", "doc_b"]
    assert results[1].retrieval_method == "graph_expanded"
    assert results[1].metadata["graph_direction"] == "downstream"


def test_graph_expansion_can_be_disabled():
    """Comparison runs can use identical base retrieval without graph candidates."""
    chunks = [Chunk(chunk_id="a_0", doc_id="doc_a", content="seed", start_char=0, end_char=4, chunk_index=0)]
    graph = KnowledgeGraph()
    retriever = GraphExpandedRetriever(StubHybridRetriever(chunks), graph)

    assert len(retriever.retrieve("seed", expand=False)) == 1