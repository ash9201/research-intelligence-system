"""
Tests for document chunking
"""
import pytest
from src.chunking import DocumentChunker


class TestDocumentChunker:
    """Test document chunking"""
    
    def test_chunk_document_recursive(self):
        """Test recursive chunking"""
        content = """
        First paragraph about topic one.
        
        Second paragraph about topic two.
        
        Third paragraph with more detailed information.
        """.strip()
        
        chunker = DocumentChunker(chunk_size=100, overlap=20, strategy="recursive")
        chunks = chunker.chunk_document(content, "doc1")
        
        assert len(chunks) > 0
        assert all(chunk.doc_id == "doc1" for chunk in chunks)
        assert all(len(chunk.content) > 0 for chunk in chunks)
    
    def test_chunk_document_sentence(self):
        """Test sentence chunking"""
        content = """
        First sentence about machine learning. Second sentence with more details.
        Third sentence with even more information. Fourth sentence concluding.
        """.strip()
        
        chunker = DocumentChunker(chunk_size=100, overlap=20, strategy="sentence")
        chunks = chunker.chunk_document(content, "doc1")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == "doc1"
            assert len(chunk.content) > 0

    def test_fixed_strategy_preserves_overlap_and_provenance(self):
        """Fixed chunking retains source metadata and produces overlapping windows."""
        content = "abcdefghijklmnopqrstuvwxyz"
        chunks = DocumentChunker(chunk_size=10, overlap=3, strategy="fixed").chunk_document(
            content,
            "doc1",
            metadata={"page": 2, "section": "Methods", "title": "Example"},
        )

        assert [chunk.content for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
        assert all(chunk.doc_id == "doc1" for chunk in chunks)
        assert all(chunk.metadata["page"] == 2 for chunk in chunks)
        assert all(chunk.metadata["section"] == "Methods" for chunk in chunks)

    def test_strategies_produce_distinct_structures(self):
        """Selectable strategies retain indices while respecting their own boundaries."""
        content = "First sentence ends here. Second sentence ends here.\n\nThird paragraph begins here."

        fixed = DocumentChunker(chunk_size=30, overlap=0, strategy="fixed").chunk_document(content, "doc1")
        sentence = DocumentChunker(chunk_size=45, overlap=0, strategy="sentence").chunk_document(content, "doc1")
        recursive = DocumentChunker(chunk_size=45, overlap=0, strategy="recursive").chunk_document(content, "doc1")

        assert [chunk.end_char for chunk in fixed] != [chunk.end_char for chunk in sentence]
        assert sentence[0].content.endswith(".")
        assert all(chunk.chunk_index == index for index, chunk in enumerate(recursive))

    def test_page_chunks_preserve_page_section_and_sentence_boundaries(self):
        """Logical page chunks retain page ranges and nearest section metadata."""
        pages = [
            {"number": 3, "text": "3.2 Attention Layers\nAttention uses queries and keys. It preserves order at the"},
            {"number": 4, "text": "bottom of the stack. 3.5 Positional Encoding\nThe model has no recurrence or convolution."},
        ]
        chunks = DocumentChunker(chunk_size=140, overlap=0, strategy="sentence").chunk_pages(pages, "paper")

        assert chunks[0].metadata["pages"] == [3, 4]
        assert chunks[0].metadata["section"] == "3.2 Attention Layers"
        assert "bottom of the stack" in chunks[0].content

    def test_logical_flow_preserves_cross_page_transformer_sections(self):
        """Cross-page prose remains coherent while page ranges and symbols survive."""
        pages = [
            {
                "number": 5,
                "text": "3.5 Positional Encoding\nSince the model has no recurrence and no convolution, it needs positional encoding at the",
            },
            {
                "number": 6,
                "text": "Table 1: Complexity results\n\nLayer Type Complexity\n\nbottom of the encoder stack so token order is available. √dₖ remains intact.",
            },
            {
                "number": 7,
                "text": "3.2.1 Scaled Dot-Product Attention\nQueries are scaled by √dₖ at the",
            },
            {
                "number": 8,
                "text": "Figure 2: Attention\n\nscaled dot-product attention boundary before softmax. warmup_steps is unchanged.",
            },
        ]
        chunks = DocumentChunker(chunk_size=220, overlap=0, strategy="recursive").chunk_pages(pages, "paper")

        positional = next(chunk for chunk in chunks if "no recurrence" in chunk.content)
        scaled = next(chunk for chunk in chunks if "scaled dot-product attention boundary" in chunk.content)

        assert positional.metadata["pages"] == [5, 6]
        assert positional.metadata["section"] == "3.5 Positional Encoding"
        assert "token order is available" in positional.content
        assert "√dₖ" in positional.content
        assert scaled.metadata["pages"] == [7, 8]
        assert "warmup_steps" in scaled.content

        layout = [chunk for chunk in chunks if chunk.metadata.get("content_type") == "layout"]
        assert any("Table 1" in chunk.content for chunk in layout)
        assert any(chunk.metadata["pages"] == [6] for chunk in layout)

    def test_page_number_and_figure_caption_do_not_replace_section_metadata(self):
        """A footer number plus a caption is not a numbered document section."""
        pages = [
            {"number": 3, "text": "3.2.1 Scaled Dot-Product Attention\nThe explanation continues at the"},
            {"number": 4, "text": "Figure 2: Attention\n\nquery boundary before softmax."},
        ]
        chunks = DocumentChunker(chunk_size=180, overlap=0, strategy="recursive").chunk_pages(pages, "paper")

        continuation = next(chunk for chunk in chunks if "query boundary" in chunk.content)
        assert continuation.metadata["section"] == "3.2.1 Scaled Dot-Product Attention"

    def test_recursive_chunking_does_not_split_technical_words(self):
        """Structure-aware chunking falls back to whitespace rather than word fragments."""
        content = "The model-dimensional representation preserves warmup_steps and √dₖ values. " * 3
        chunks = DocumentChunker(chunk_size=90, overlap=0, strategy="recursive").chunk_document(content, "doc1")

        assert all(not chunk.content.startswith("imensional") for chunk in chunks)
        assert all(not chunk.content.endswith("model-") for chunk in chunks)
        assert any("warmup_steps" in chunk.content for chunk in chunks)
    
    def test_chunks_have_correct_indices(self):
        """Test that chunks have correct indices"""
        content = "Chunk one. Chunk two. Chunk three. Chunk four. Chunk five."
        
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_document(content, "doc1")
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
