"""Regression tests for logical cross-page chunking of the bundled Transformer paper."""
from pathlib import Path

import pytest

from src.chunking import DocumentChunker
from src.ingestion import DocumentLoader


PAPER_PATH = Path("data/documents/attention-is-all-you-need.pdf")


@pytest.fixture(scope="module")
def attention_chunks():
    """Create the current logical-flow chunks directly from the real paper fixture."""
    if not PAPER_PATH.exists():
        pytest.skip("Attention Is All You Need PDF fixture is not available")
    document = DocumentLoader.load_document(PAPER_PATH)
    return DocumentChunker(chunk_size=512, overlap=128, strategy="recursive").chunk_pages(
        document.metadata["pages"],
        document.doc_id,
        {"title": document.title},
    )


def test_attention_cross_page_sections_remain_coherent(attention_chunks):
    """The paper's interrupted prose is reconstructed across physical page boundaries."""
    scaled = next(
        chunk
        for chunk in attention_chunks
        if (chunk.metadata.get("section") or "").startswith("3.2.1")
        and "divide each by √dk" in chunk.content
    )
    positional = next(
        chunk
        for chunk in attention_chunks
        if (chunk.metadata.get("section") or "").startswith("3.5")
        and "no recurrence and no convolution" in chunk.content
    )
    multi_page_multi_head = next(
        chunk
        for chunk in attention_chunks
        if (chunk.metadata.get("section") or "").startswith("3.2.2")
        and chunk.metadata["pages"] == [4, 5]
    )

    assert scaled.metadata["pages"] == [3, 4]
    assert "query with all keys" in scaled.content
    assert positional.metadata["pages"] == [5, 6]
    assert "bottoms of the encoder and decoder stacks" in positional.content
    assert "dmodel" in positional.content
    assert "MultiHead" in multi_page_multi_head.content


def test_attention_layout_isolated_from_semantic_prose(attention_chunks):
    """Figure/table text is preserved in layout chunks rather than breaking prose chunks."""
    layout_chunks = [chunk for chunk in attention_chunks if chunk.metadata.get("content_type") == "layout"]

    assert any("Figure 2" in chunk.content for chunk in layout_chunks)
    assert any("Table 1" in chunk.content for chunk in layout_chunks)
    assert all(chunk.metadata["pages"] == [chunk.metadata["page"]] for chunk in layout_chunks)